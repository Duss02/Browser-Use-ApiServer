# Goal: Checks for available visa appointment slots on the Greece MFA website.

import asyncio
import os
import uvicorn
import threading
import json
import logging
from enum import Enum
from typing import Optional

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

class ActionRequest(BaseModel):
    """Model for action execution request."""
    action_type: ActionType  # Using Enum for better validation
    element_id: Optional[str] = None  # ID dell'elemento se disponibile
    element_label: Optional[str] = None  # Testo/etichetta dell'elemento
    value: Optional[str] = None  # Valore da inserire per input/select
    
    @root_validator(pre=True)
    def check_element_identifiers(cls, values):
        """Validate that at least one element identifier is provided."""
        element_id = values.get('element_id')
        element_label = values.get('element_label')
        
        if not element_id and not element_label:
            raise ValueError("At least one of element_id or element_label must be provided")
        
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

async def get_valid_page():
    """Helper function to get a valid browser page, handling closed browser cases."""
    global browser
    try:
        # We need to create a new context and page since the API doesn't support getting current
        try:
            # Create a new browser context
            browser_context = await browser.new_context()
            # Get a page from this context
            page = await browser_context.get_agent_current_page()
            logger.info("Created new browser context and page")
            return page, browser_context
        except Exception as e:
            logger.error(f"Failed to create context or page: {str(e)}")
            
            # Try reinitializing the browser
            try:
                # Reinitialize the browser
                browser = Browser(
                    config=BrowserConfig(
                        browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        extra_browser_args=[
                            "--user-data-dir=remote-debug-profile",
                        ],
                        headless=False,
                    )
                )
                # Create a new context and page
                browser_context = await browser.new_context()
                page = await browser_context.get_agent_current_page()
                await page.goto(WebpageInfo().link)  # Navigate to default page
                logger.info("Recreated browser and opened new page")
                return page, browser_context
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize browser: {str(reinit_error)}")
                raise
    except Exception as final_error:
        logger.error(f"Failed to get valid page after all attempts: {str(final_error)}")
        raise

@app.post("/analyze-webpage")
async def analyze_webpage(webpage_info: WebpageInfo):
    """Endpoint that analyzes a webpage using the browser agent."""
    try:
        # Get a valid page, possibly creating a new context if needed
        page, browser_context = await get_valid_page()
        
        # Navigate to the specified URL
        await page.goto(webpage_info.link)
        
        task = """Analyze the webpage and in the open tab list all the elements that users can engage with. Only show the viewable elements:

1. For clickable elements (buttons, links, etc.):
{
  "clickElements": [
    {
      "id": "string or null if not available",
      "label": "text displayed on the element",
      "description": "brief description of what this element does"
    }
  ]
}

2. For selection elements (dropdowns, radio groups, etc.):
{
  "selectElements": [
    {
      "id": "string or null if not available",
      "label": "text associated with this selection element",
      "description": "what this selection controls or affects",
      "options": ["option1", "option2", "option3"]
    }
  ]
}

3. For input fields:
{
  "inputElements": [
    {
      "id": "string or null if not available",
      "label": "field label text",
      "description": "what information this field collects",
      "placeholder": "placeholder text if present, otherwise empty string",
      "type": "text, number, email, or password"
    }
  ]
}

Ensure you:
- Include only elements that are currently visible and interactive
- Accurately capture all available options for select elements
- Determine the correct input type (text/number/email/password)
- Use the exact element text for labels
- Provide meaningful descriptions of each element's purpose
- Return null for missing IDs rather than omitting the field

The output must be valid JSON that strictly follows this structure.
DO NOT CLICK ON ANY ELEMENTS, JUST ANALYZE THE"""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = Agent(task, model, controller=controller, use_vision=True, browser=browser)
        
        # Run the agent and capture the result
        result = await agent.run()
        
        # Get the final result
        contenuto_estratto = result.final_result()
        
        # Close the browser context only if we created a new one
        if browser_context:
            await browser_context.close()
        
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

@app.post("/execute-action")
async def execute_action(request: ActionRequest):
    """Endpoint that executes a specific action extracted previously on the current page."""
    try:
        # Get a valid page, possibly creating a new context if needed
        page, browser_context = await get_valid_page()
        
        # Build a description of the element based on all available details
        element_identifiers = []
        
        if request.element_id and request.element_id != "null":
            element_identifiers.append(f'ID "{request.element_id}"')
            
        if request.element_label:
            element_identifiers.append(f'text or label "{request.element_label}"')
            
        if not element_identifiers:
            raise HTTPException(status_code=400, detail="No identifier provided for the element")
            
        element_description = " or ".join(element_identifiers)
        
        # Build the prompt based on the action type
        action_prompt = ""
        
        if request.action_type == ActionType.CLICK:
            action_prompt = f"""On the currently open web page,
            find and click on the element with {element_description}.
            
            Detailed instructions:
            1. Carefully search for the element that matches the provided criteria
            2. If you find more than one matching element, use context and position to choose the most likely one
            3. Execute a click on that element
            4. Describe in detail what happened after the click (new page, popup, change on the page, etc.)
            
            IMPORTANT: If the ID is not available, focus on the element's text or other visual identifiers."""
            
        elif request.action_type == ActionType.SELECT:
            action_prompt = f"""On the currently open web page,
            find the selector with {element_description} and select the option "{request.value}".
            
            Detailed instructions:
            1. Carefully search for the selector that matches the provided criteria
            2. After finding it, select the option "{request.value}" from the menu
            3. If the exact option doesn't exist, choose the most similar one
            4. Describe the result of the selection and any changes on the page
            
            IMPORTANT: If the ID is not available, focus on the selector's label or other visual identifiers."""
            
        elif request.action_type == ActionType.INPUT:
            action_prompt = f"""On the currently open web page,
            find the input field with {element_description} and enter the value "{request.value}".
            
            Detailed instructions:
            1. Carefully search for the input field that matches the provided criteria
            2. Click on the field to activate it
            3. Enter exactly this value: "{request.value}"
            4. DO NOT press enter or submit the form after input, unless specifically requested
            5. Describe the state of the field after input and any changes on the page
            
            IMPORTANT: If the ID is not available, focus on the field's label, placeholder, or other visual identifiers."""
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action type: {request.action_type}")
        
        # Execute the action
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = Agent(action_prompt, model, controller=controller, use_vision=True, browser=browser)
        result = await agent.run()
        
        # Get the action result
        action_result = result.final_result()
        
        # Don't close the browser context even if we created a new one - keep it for future use
        
        if not action_result:
            return {"success": False, "message": "No result obtained from executing the action"}
        
        return {
            "success": True,
            "action_type": request.action_type,
            "element_description": element_description,
            "value": request.value,
            "result": action_result
        }
            
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
	
	task = """Analyze the webpage and in the open tab list all the elements that users can engage with. Only show the viewable elements:

1. For clickable elements (buttons, links, etc.):
{
  "clickElements": [
    {
      "id": "string or null if not available",
      "label": "text displayed on the element",
      "description": "brief description of what this element does"
    }
  ]
}

2. For selection elements (dropdowns, radio groups, etc.):
{
  "selectElements": [
    {
      "id": "string or null if not available",
      "label": "text associated with this selection element",
      "description": "what this selection controls or affects",
      "options": ["option1", "option2", "option3"]
    }
  ]
}

3. For input fields:
{
  "inputElements": [
    {
      "id": "string or null if not available",
      "label": "field label text",
      "description": "what information this field collects",
      "placeholder": "placeholder text if present, otherwise empty string",
      "type": "text, number, email, or password"
    }
  ]
}

Ensure you:
- Include only elements that are currently visible and interactive
- Accurately capture all available options for select elements
- Determine the correct input type (text/number/email/password)
- Use the exact element text for labels
- Provide meaningful descriptions of each element's purpose
- Return null for missing IDs rather than omitting the field

The output must be valid JSON that strictly follows this structure.
DO NOT CLICK ON ANY ELEMENTS, JUST ANALYZE THE"""

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





