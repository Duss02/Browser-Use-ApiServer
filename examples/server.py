# Goal: Checks for available visa appointment slots on the Greece MFA website.

import asyncio
import os
import uvicorn
import threading
import json
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from pydantic import BaseModel, SecretStr, validator, root_validator

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller
from browser_use import BrowserConfig
from browser_use import Browser

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS to allow all origins, methods and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebpageInfo(BaseModel):
	"""Model for webpage link."""

	link: str = 'https://google.com'

class ActionType(str, Enum):
    """Enum for action types"""
    CLICK = "click"
    SELECT = "select"
    INPUT = "input"
    FORM = "form"

class FormInput(BaseModel):
    """Model for a single input in a form."""
    field_id: Optional[str] = None
    field_label: Optional[str] = None
    value: str

class ActionRequest(BaseModel):
    """Model for action execution request."""
    action_type: ActionType  # Using Enum for better validation
    element_id: Optional[str] = None  # ID dell'elemento se disponibile
    element_label: Optional[str] = None  # Testo/etichetta dell'elemento
    value: Optional[str] = None  # Valore da inserire per input/select
    form_inputs: Optional[List[FormInput]] = None  # Lista di input per form
    submit_form: Optional[bool] = True  # Se inviare il form dopo aver compilato i campi
    
    @root_validator(pre=True)
    def check_element_identifiers(cls, values):
        """Validate that at least one element identifier is provided."""
        action_type = values.get('action_type')
        element_id = values.get('element_id')
        element_label = values.get('element_label')
        form_inputs = values.get('form_inputs')
        
        if action_type != ActionType.FORM and not element_id and not element_label:
            raise ValueError("At least one of element_id or element_label must be provided")
            
        if action_type == ActionType.FORM and not form_inputs:
            raise ValueError("form_inputs must be provided for form actions")
        
        return values
    
    @validator('value')
    def value_required_for_input_and_select(cls, v, values):
        """Validate that value is provided for input and select actions."""
        action_type = values.get('action_type')
        if action_type in [ActionType.INPUT, ActionType.SELECT] and not v:
            raise ValueError(f"Value must be provided for action type '{action_type}'")
        return v

@app.get("/")
def read_root():
    return {"Hello": "World"}

def estrai_json_da_llm(testo):
    """Estrae il JSON puro da un output di LLM che potrebbe contenere delimitatori di codice markdown."""
    if not testo:
        return None
    
    # Se il testo contiene un blocco di codice markdown JSON
    if "```json" in testo:
        # Trova l'inizio e la fine del blocco JSON
        inizio = testo.find("```json")
        if inizio != -1:
            inizio = testo.find("\n", inizio) + 1  # Salta la riga con ```json
            fine = testo.find("```", inizio)
            if fine != -1:
                json_testo = testo[inizio:fine].strip()
                try:
                    return json.loads(json_testo)
                except json.JSONDecodeError:
                    return None
    
    # Prova a vedere se l'intero testo è JSON valido
    try:
        return json.loads(testo)
    except json.JSONDecodeError:
        pass
    
    return None

def salva_json_pulito(json_data, percorso_file):
    """Salva solo il JSON pulito in un file."""
    
    try:
        with open(percorso_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print("Errore nel salvare il JSON pulito", e)
        return False

# Global variables for browser context
global_browser_context = None

# Simplified function to get a valid page
async def get_valid_page():
    """Simple helper that always creates a new context and page."""
    global browser
    
    try:
        # Create a new context and page
        context = await browser.new_context()
        page = await context.get_agent_current_page()
        
        # Check if the page is on localhost:3000
        if "localhost:3000" in page.url:
            logger.info("Found page on localhost:3000, creating a new page")
            try:
                # Create a new page in the same context
                new_page = await context.new_page()
                await new_page.goto("https://google.com")  # Navigate to a default page
                logger.info("Created new page to avoid localhost:3000")
                return new_page, context
            except Exception as e:
                # If creating a new page fails, close this context and create a brand new one
                logger.warning(f"Error creating new page: {str(e)}")
                await context.close()
                
                # Create another fresh context and page
                new_context = await browser.new_context()
                new_page = await new_context.new_page()
                await new_page.goto("https://google.com")
                logger.info("Created new context and page to avoid localhost:3000")
                return new_page, new_context
        
        logger.info("Created new browser context and page")
        return page, context
    except Exception as e:
        logger.error(f"Error creating context or page: {str(e)}")
        
        # Reinitialize the browser and try again
        try:
            logger.info("Reinitializing browser and trying again")
            browser = Browser(
                config=BrowserConfig(
                    browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    extra_browser_args=[
                        "--user-data-dir=remote-debug-profile",
                    ],
                    headless=False,
                )
            )
            context = await browser.new_context()
            page = await context.get_agent_current_page()
            
            # Check again for localhost:3000
            if "localhost:3000" in page.url:
                logger.info("Found page on localhost:3000 after reinitialization, creating a new page")
                new_page = await context.new_page()
                await new_page.goto("https://google.com")
                logger.info("Created new page to avoid localhost:3000")
                return new_page, context
                
            logger.info("Successfully created page after browser reinitialization")
            return page, context
        except Exception as reinit_error:
            logger.critical(f"Critical error: Failed to get page even after browser reinitialization: {str(reinit_error)}")
            raise Exception("Could not get a valid page or browser after multiple attempts")

# Cache for page analysis results
# Structure: {url: {"timestamp": datetime, "data": analysis_result}}
page_analysis_cache: Dict[str, Dict[str, Any]] = {}
# Cache TTL in minutes
CACHE_TTL_MINUTES = 5

def invalidate_cache(url: str = None):
    """Invalidate the cache for a specific URL or all URLs."""
    global page_analysis_cache
    if url:
        if url in page_analysis_cache:
            logger.info(f"Invalidating cache for URL: {url}")
            page_analysis_cache.pop(url)
    else:
        logger.info("Invalidating entire cache")
        page_analysis_cache.clear()

def get_cached_analysis(url: str):
    """Get cached analysis for a URL if available and not expired."""
    if url in page_analysis_cache:
        cache_entry = page_analysis_cache[url]
        # Check if cache is still valid
        if datetime.now() - cache_entry["timestamp"] < timedelta(minutes=CACHE_TTL_MINUTES):
            logger.info(f"Cache hit for URL: {url}")
            return cache_entry["data"]
        else:
            logger.info(f"Cache expired for URL: {url}")
            page_analysis_cache.pop(url)
    return None

def cache_analysis(url: str, data: Any):
    """Cache analysis data for a URL."""
    global page_analysis_cache
    logger.info(f"Caching analysis for URL: {url}")
    page_analysis_cache[url] = {
        "timestamp": datetime.now(),
        "data": data
    }

@app.post("/analyze-webpage")
async def analyze_webpage(webpage_info: WebpageInfo):
    """Endpoint that analyzes a webpage using the browser agent."""
    try:
        # Get a fresh page and context
        page, context = await get_valid_page()
        
        # Navigate to the specified URL
        try:
            await page.goto(webpage_info.link)
        except Exception as nav_error:
            logger.error(f"Navigation error: {str(nav_error)}")
            await context.close()
            raise HTTPException(status_code=400, detail=f"Failed to navigate to {webpage_info.link}: {str(nav_error)}")
        
        task = """Analyze the webpage and in the open tab list all the elements that users can engage with. Only show the viewable elements.

Return a single JSON array of elements with the following structure:

{
  "elements": [
    {
      "type": "click", 
      "id": "string or null if not available",
      "label": "text displayed on the element",
      "description": "brief description of what this element does",
      "importance": 8 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "select",
      "id": "string or null if not available",
      "label": "text associated with this selection element",
      "description": "what this selection controls or affects",
      "options": ["option1", "option2", "option3"],
      "importance": 7 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "input",
      "id": "string or null if not available",
      "label": "field label text",
      "description": "what information this field collects",
      "placeholder": "placeholder text if present, otherwise empty string",
      "inputType": "text, number, email, or password",
      "importance": 9 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "form",
      "id": "string or null if not available",
      "description": "what purpose this form serves",
      "submitButton": {
        "id": "string or null if not available",
        "label": "text on the submit button"
      },
      "inputs": [
        {
          "id": "string or null if not available",
          "label": "field label text",
          "description": "what information this field collects",
          "placeholder": "placeholder text if present, otherwise empty string",
          "inputType": "text, number, email, or password",
          "required": true or false
        }
      ],
      "importance": 10 // Forms are typically high importance elements
    }
  ]
}

IMPORTANT INSTRUCTIONS FOR FORMS:
- Actively look for elements that should be grouped together into forms
- Always group search inputs with their search buttons into a form
- If a search input exists without a visible button, still create a form and add a dummy submit button with label "Search"
- For login forms, group username/email and password fields together
- Group related inputs like address fields (street, city, state, zip) into a single form
- If a set of radio buttons or checkboxes appear related, group them into a form
- ANY input field that submits data when pressing Enter should be considered a form, even if no visible submit button exists

Ensure you:
- Include only elements that are currently visible and interactive
- Identify forms and group their inputs together with the submit button
- Evaluate importance based on: prominence on page, typical user goals, and whether the element is required
- Sort elements by importance (highest to lowest)
- Accurately capture all available options for select elements
- Determine the correct input type (text/number/email/password) for input fields
- Use the exact element text for labels
- Provide meaningful descriptions of each element's purpose
- Return null for missing IDs rather than omitting the field

The output must be valid JSON that strictly follows this structure.
DO NOT CLICK ON ANY ELEMENTS, JUST ANALYZE THE PAGE."""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = Agent(task, model, controller=controller, use_vision=True, browser=browser)
        
        # Run the agent and capture the result
        result = await agent.run()
        
        # Get the final result
        contenuto_estratto = result.final_result()
        
        # Always close the context when done
        await context.close()
        
        if not contenuto_estratto:
            return {"success": False, "message": "No result obtained from analysis"}
        
        # Extract clean JSON from the result
        json_data = estrai_json_da_llm(contenuto_estratto)
        logger.info(f"Raw result: {contenuto_estratto[:500]}...")  # Log the first 500 chars of the raw result
        
        if json_data:
            return {"success": True, "data": json_data}
        else:
            # Try to fix common JSON issues before giving up
            fixed_result = fix_json_format(contenuto_estratto)
            json_data = estrai_json_da_llm(fixed_result)
            
            if json_data:
                return {"success": True, "data": json_data}
            else:
                return {"success": False, "message": "Unable to extract valid JSON from result", "raw": contenuto_estratto}
            
    except Exception as e:
        logger.error(f"Error in analyze-webpage: {str(e)}")
        return {"success": False, "error": str(e)}

def fix_json_format(text):
    """Attempt to fix common JSON formatting issues."""
    # Try to extract JSON if it's embedded in other text
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < end:
            text = text[start:end]
    
    # Replace single quotes with double quotes (common LLM mistake)
    text = text.replace("'", '"')
    
    # Ensure property names are double-quoted
    import re
    text = re.sub(r'([{,])\s*(\w+):', r'\1 "\2":', text)
    
    return text

@app.post("/analyze-current-page")
async def analyze_current_page():
    """Endpoint that analyzes the currently open page using the browser agent."""
    try:
        # Get a fresh page and context
        page, context = await get_valid_page()
        
        # Get the current URL
        current_url = page.url
        
        # Check if we have a cached result for this URL
        cached_result = get_cached_analysis(current_url)
        if cached_result:
            # Always close the context when done
            await context.close()
            return {"success": True, "data": cached_result, "cached": True}
        
        # Run the analysis on the current page
        task = """Analyze the webpage and in the open tab list all the elements that users can engage with. Only show the viewable elements.

Return a single JSON array of elements with the following structure:

{
  "elements": [
    {
      "type": "click", 
      "id": "string or null if not available",
      "label": "text displayed on the element",
      "description": "brief description of what this element does",
      "importance": 8 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "select",
      "id": "string or null if not available",
      "label": "text associated with this selection element",
      "description": "what this selection controls or affects",
      "options": ["option1", "option2", "option3"],
      "importance": 7 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "input",
      "id": "string or null if not available",
      "label": "field label text",
      "description": "what information this field collects",
      "placeholder": "placeholder text if present, otherwise empty string",
      "inputType": "text, number, email, or password",
      "importance": 9 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "form",
      "id": "string or null if not available",
      "description": "what purpose this form serves",
      "submitButton": {
        "id": "string or null if not available",
        "label": "text on the submit button"
      },
      "inputs": [
        {
          "id": "string or null if not available",
          "label": "field label text",
          "description": "what information this field collects",
          "placeholder": "placeholder text if present, otherwise empty string",
          "inputType": "text, number, email, or password",
          "required": true or false
        }
      ],
      "importance": 10 // Forms are typically high importance elements
    }
  ]
}

IMPORTANT INSTRUCTIONS FOR FORMS:
- Actively look for elements that should be grouped together into forms
- Always group search inputs with their search buttons into a form
- If a search input exists without a visible button, still create a form and add a dummy submit button with label "Search"
- For login forms, group username/email and password fields together
- Group related inputs like address fields (street, city, state, zip) into a single form
- If a set of radio buttons or checkboxes appear related, group them into a form
- ANY input field that submits data when pressing Enter should be considered a form, even if no visible submit button exists

Ensure you:
- Include only elements that are currently visible and interactive
- Identify forms and group their inputs together with the submit button
- Evaluate importance based on: prominence on page, typical user goals, and whether the element is required
- Sort elements by importance (highest to lowest)
- Accurately capture all available options for select elements
- Determine the correct input type (text/number/email/password) for input fields
- Use the exact element text for labels
- Provide meaningful descriptions of each element's purpose
- Return null for missing IDs rather than omitting the field

The output must be valid JSON that strictly follows this structure.
DO NOT CLICK ON ANY ELEMENTS, JUST ANALYZE THE PAGE."""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = Agent(task, model, controller=controller, use_vision=True, browser=browser)
        
        # Run the agent and capture the result
        result = await agent.run()
        
        # Get the final result
        contenuto_estratto = result.final_result()
        
        # Always close the context when done
        await context.close()
        
        if not contenuto_estratto:
            return {"success": False, "message": "No result obtained from analysis"}
        
        # Extract clean JSON from the result
        json_data = estrai_json_da_llm(contenuto_estratto)
        logger.info(f"Raw result: {contenuto_estratto[:500]}...")  # Log the first 500 chars of the raw result
        
        if not json_data:
            # Try to fix common JSON issues before giving up
            fixed_result = fix_json_format(contenuto_estratto)
            json_data = estrai_json_da_llm(fixed_result)
        
        if json_data:
            # Cache the result
            cache_analysis(current_url, json_data)
            return {"success": True, "data": json_data, "cached": False}
        else:
            return {"success": False, "message": "Unable to extract valid JSON from result", "raw": contenuto_estratto}
            
    except Exception as e:
        logger.error(f"Error in analyze-current-page: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/execute-action")
async def execute_action(request: ActionRequest):
    """Endpoint that executes a specific action extracted previously on the current page."""
    try:
        # Get a fresh page and context
        page, context = await get_valid_page()
        
        # Get the current URL before the action
        current_url = page.url
        
        # Build the prompt based on the action type
        action_prompt = ""
        
        if request.action_type == ActionType.CLICK:
            # Build a description of the element based on all available details
            element_identifiers = []
            
            if request.element_id and request.element_id != "null":
                element_identifiers.append(f'ID "{request.element_id}"')
                
            if request.element_label:
                element_identifiers.append(f'text or label "{request.element_label}"')
                
            if not element_identifiers:
                raise HTTPException(status_code=400, detail="No identifier provided for the element")
                
            element_description = " or ".join(element_identifiers)
            
            action_prompt = f"""On the currently open web page,
            find and click on the element with {element_description}.
            
            Detailed instructions:
            1. Carefully search for the element that matches the provided criteria
            2. If you find more than one matching element, use context and position to choose the most likely one
            3. Execute a click on that element
            4. Describe in detail what happened after the click (new page, popup, change on the page, etc.)
            
            IMPORTANT: 
            - If the ID is not available, focus on the element's text or other visual identifiers
            - ONLY perform the single click action described above and nothing more
            - DO NOT continue with any additional actions after the click unless explicitly instructed"""
            
        elif request.action_type == ActionType.SELECT:
            # Build a description of the element based on all available details
            element_identifiers = []
            
            if request.element_id and request.element_id != "null":
                element_identifiers.append(f'ID "{request.element_id}"')
                
            if request.element_label:
                element_identifiers.append(f'text or label "{request.element_label}"')
                
            if not element_identifiers:
                raise HTTPException(status_code=400, detail="No identifier provided for the element")
                
            element_description = " or ".join(element_identifiers)
            
            action_prompt = f"""On the currently open web page,
            find the selector with {element_description} and select the option "{request.value}".
            
            Detailed instructions:
            1. Carefully search for the selector that matches the provided criteria
            2. After finding it, select the option "{request.value}" from the menu
            3. If the exact option doesn't exist, choose the most similar one
            4. Describe the result of the selection and any changes on the page
            
            IMPORTANT: 
            - If the ID is not available, focus on the selector's label or other visual identifiers
            - ONLY perform the single select action described above and nothing more
            - DO NOT continue with any additional actions after the selection unless explicitly instructed"""
            
        elif request.action_type == ActionType.INPUT:
            # Build a description of the element based on all available details
            element_identifiers = []
            
            if request.element_id and request.element_id != "null":
                element_identifiers.append(f'ID "{request.element_id}"')
                
            if request.element_label:
                element_identifiers.append(f'text or label "{request.element_label}"')
                
            if not element_identifiers:
                raise HTTPException(status_code=400, detail="No identifier provided for the element")
                
            element_description = " or ".join(element_identifiers)
            
            action_prompt = f"""On the currently open web page,
            find the input field with {element_description} and enter the value "{request.value}".
            
            Detailed instructions:
            1. Carefully search for the input field that matches the provided criteria
            2. Click on the field to activate it
            3. Enter exactly this value: "{request.value}"
            4. DO NOT press enter or submit the form after input, unless specifically requested
            5. Describe the state of the field after input and any changes on the page
            
            IMPORTANT: 
            - If the ID is not available, focus on the field's label, placeholder, or other visual identifiers
            - ONLY perform the single input action described above and nothing more
            - DO NOT continue with any additional actions after entering the value unless explicitly instructed"""
            
        elif request.action_type == ActionType.FORM:
            # Form filling requires a different approach
            form_inputs_json = json.dumps([input_model.dict() for input_model in request.form_inputs])
            
            # Determine if this is likely a search form
            is_search_form = False
            search_terms = ["search", "cerca", "find", "query", "q", "s", "keyword", "buscar", "recherche"]
            
            # Check input fields and their descriptions for search terms
            for input_item in request.form_inputs:
                field_id = input_item.field_id or ""
                field_label = input_item.field_label or ""
                
                # Check if any search terms appear in the field ID or label
                if any(term.lower() in field_id.lower() for term in search_terms) or \
                   any(term.lower() in field_label.lower() for term in search_terms):
                    is_search_form = True
                    break
            
            if request.submit_form:
                if is_search_form:
                    submit_instructions = """After filling the search field, you have two options to submit the search:
                    
                    Option 1: Press the ENTER key while the search field is active
                    - After typing the search term, press Enter on the keyboard
                    - This is often more reliable for search forms
                    
                    Option 2: Click the search button if visible
                    - Look for a button with a magnifying glass icon
                    - Or a button with text like "Search", "Go", "Find", etc.
                    - The button is typically next to the search field
                    
                    Try Option 1 (pressing Enter) first, and only if that doesn't work, try Option 2 (clicking the button).
                    """
                else:
                    submit_instructions = """After filling all fields, find and CLICK the submit button of the form. 
                    Look for:
                    - A button with text like "Submit", "Send", "Login", "Sign In", "Continue", etc.
                    - A button near the form fields that appears to be the primary action
                    - An icon button that indicates submission
                    
                    You MUST physically click the submit button, not just press Enter or call form.submit().
                    """
            else:
                submit_instructions = "After filling all fields, DO NOT submit the form or click any buttons."
            
            action_prompt = f"""On the currently open web page,
            fill out a form with the following inputs:
            
            {form_inputs_json}
            
            Detailed instructions:
            1. For each input in the list:
               a. Find the field using its ID or label
               b. Click on the field
               c. Enter the specified value
            2. {submit_instructions}
            
            When identifying form fields:
            - For fields with ID, use that as the primary identifier
            - For fields with no ID, use the label text
            - Look for labels near the input fields
            - Consider placeholder text as a fallback identifier
            
            IMPORTANT:
            - Fill ONLY the fields specified in the list
            - {"For search forms, prefer pressing ENTER in the search field over clicking buttons" if is_search_form else "If instructed to submit, you MUST find and CLICK the actual submit button"}
            - DO NOT continue with any additional actions after the form interaction
            - Report exactly what you did and the result, including how you submitted the form"""
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action type: {request.action_type}")
        
        # Execute the action
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = Agent(action_prompt, model, controller=controller, use_vision=True, browser=browser)
        result = await agent.run()
        
        # Get the action result
        action_result = result.final_result()
        
        # Get the URL after the action
        new_url = page.url
        
        # Invalidate cache for the page since it might have changed
        invalidate_cache(current_url)
        
        # If we navigated to a new page, invalidate that cache too
        if new_url != current_url:
            invalidate_cache(new_url)
        
        # Always close the context when done
        await context.close()
        
        if not action_result:
            return {"success": False, "message": "No result obtained from executing the action"}
        
        # Prepare the response based on action type
        response = {
            "success": True,
            "action_type": request.action_type,
            "result": action_result,
            "url_changed": new_url != current_url,
            "new_url": new_url if new_url != current_url else None
        }
        
        # Add action-specific details to the response
        if request.action_type == ActionType.CLICK:
            element_description = " or ".join(element_identifiers)
            response["element_description"] = element_description
        elif request.action_type == ActionType.SELECT:
            element_description = " or ".join(element_identifiers)
            response["element_description"] = element_description
            response["value"] = request.value
        elif request.action_type == ActionType.INPUT:
            element_description = " or ".join(element_identifiers)
            response["element_description"] = element_description
            response["value"] = request.value
        elif request.action_type == ActionType.FORM:
            response["form_inputs"] = [input_model.dict() for input_model in request.form_inputs]
            response["submitted"] = request.submit_form
        
        return response
            
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in execute-action: {str(e)}")
        return {"success": False, "error": str(e)}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in execute-action: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/current-page")
async def get_current_page():
    """Endpoint that returns information about the currently open page."""
    try:
        # Get a fresh page and context
        page, context = await get_valid_page()
        
        # Get only the URL of the current page
        url = page.url
        
        # Always close the context when done
        await context.close()
        
        return {
            "success": True,
            "page_info": {
                "url": url
            }
        }
            
    except Exception as e:
        logger.error(f"Error in get-current-page: {str(e)}")
        return {"success": False, "error": str(e)}

# Load environment variables
load_dotenv()
if not os.getenv('GEMINI_API_KEY'):
	raise ValueError('GEMINI_API_KEY is not set. Please add it to your environment variables.')

controller = Controller()

# Reuse existing browser
browser = Browser(
    config=BrowserConfig(
        browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        extra_browser_args=[
            "--user-data-dir=remote-debug-profile",
        ],
        headless=False,
    )
)

async def browser_agent_task():
	"""Main function to execute the agent task."""
	
	task = """Analyze the webpage and in the open tab and list all the elements that users can engage with. Only show the viewable elements.

Return a single JSON array of elements with the following structure:

{
  "elements": [
    {
      "type": "click", 
      "id": "string or null if not available",
      "label": "text displayed on the element",
      "description": "brief description of what this element does",
      "importance": 8 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "select",
      "id": "string or null if not available",
      "label": "text associated with this selection element",
      "description": "what this selection controls or affects",
      "options": ["option1", "option2", "option3"],
      "importance": 7 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "input",
      "id": "string or null if not available",
      "label": "field label text",
      "description": "what information this field collects",
      "placeholder": "placeholder text if present, otherwise empty string",
      "inputType": "text, number, email, or password",
      "importance": 9 // On a scale from 1-10, how important this element is for users
    },
    {
      "type": "form",
      "id": "string or null if not available",
      "description": "what purpose this form serves",
      "submitButton": {
        "id": "string or null if not available",
        "label": "text on the submit button"
      },
      "inputs": [
        {
          "id": "string or null if not available",
          "label": "field label text",
          "description": "what information this field collects",
          "placeholder": "placeholder text if present, otherwise empty string",
          "inputType": "text, number, email, or password",
          "required": true or false
        }
      ],
      "importance": 10 // Forms are typically high importance elements
    }
  ]
}

IMPORTANT INSTRUCTIONS FOR FORMS:
- Actively look for elements that should be grouped together into forms
- Always group search inputs with their search buttons into a form
- If a search input exists without a visible button, still create a form and add a dummy submit button with label "Search"
- For login forms, group username/email and password fields together
- Group related inputs like address fields (street, city, state, zip) into a single form
- If a set of radio buttons or checkboxes appear related, group them into a form
- ANY input field that submits data when pressing Enter should be considered a form, even if no visible submit button exists

Ensure you:
- Include only elements that are currently visible and interactive
- Identify forms and group their inputs together with the submit button
- Evaluate importance based on: prominence on page, typical user goals, and whether the element is required
- Sort elements by importance (highest to lowest)
- Accurately capture all available options for select elements
- Determine the correct input type (text/number/email/password) for input fields
- Use the exact element text for labels
- Provide meaningful descriptions of each element's purpose
- Return null for missing IDs rather than omitting the field

The output must be valid JSON that strictly follows this structure.
DO NOT CLICK ON ANY ELEMENTS, JUST ANALYZE THE PAGE."""

	model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
	# Create a new browser context and get a page
	browser_context = await browser.new_context()
	page = await browser_context.get_agent_current_page()
	await page.goto(WebpageInfo().link)
	agent = Agent(task, model, controller=controller, use_vision=True, browser=browser)

	# Run the agent and capture the result
	result = await agent.run()
	return result

def run_fastapi():
    """Run the FastAPI server with uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

def analizza_risultato(risultato):
    """Analyze the browser agent's result and extract useful information."""
    print("\n=== RESULT ANALYSIS ===")
    
    # Number of steps executed
    print(f"Total steps: {risultato.number_of_steps()}")
    
    # Total duration in seconds
    print(f"Total duration: {risultato.total_duration_seconds():.2f} seconds")
    
    # Input tokens used
    print(f"Total input tokens: {risultato.total_input_tokens()}")
    
    # URLs visited
    print(f"URLs visited: {risultato.urls()}")
    
    # Check for errors
    if risultato.has_errors():
        print("Errors encountered:")
        for error in risultato.errors():
            if error:
                print(f"  - {error}")
    else:
        print("No errors encountered")
    
    # Get the final result
    risultato_finale = risultato.final_result()
    if risultato_finale:
        print("\nFinal result:")
        print(risultato_finale)
    
    # All actions executed
    print("\nActions executed:")
    for action in risultato.action_names():
        print(f"  - {action}")
    
    return risultato_finale

async def main():
    """Run both FastAPI and browser agent concurrently."""
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True  # Thread will exit when main program exits
    fastapi_thread.start()
    
    # # Run browser agent and capture the result
    # result = await browser_agent_task()
    
    # # Use the result, for example by printing it
    # print("Browser agent result:")
    # print(result)
    
    # # Detailed analysis of the result
    # risultato_dettagliato = analizza_risultato(result)
    
    # # Extract and save only the clean JSON
    # contenuto_estratto = result.final_result()
    # if contenuto_estratto:
    #     json_data = estrai_json_da_llm(contenuto_estratto)
    #     if json_data:
    #         # Save only the clean JSON
    #         salva_json_pulito(json_data, "risultato_json_pulito.json")
    #         print("\nClean JSON extracted and saved to 'risultato_json_pulito.json'")
    #     else:
    #         print("\nUnable to extract valid JSON from result")
    
    # # You can access various methods and properties of AgentHistoryList
    # if result.is_done():
    #     print("The agent has completed its task")
    
    # # Keep the main program running if needed
    await asyncio.Event().wait()

if __name__ == '__main__':
	asyncio.run(main())





