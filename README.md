Here is a README file tailored for your `server.py` file, including instructions for setting up `uvicorn` and configuring the environment variables.

# Link client
https://github.com/Duss02/app-gdg

---

# Webpage Interaction and Analysis Server

This project provides a FastAPI-based web server designed to analyze and interact with web pages. The `server.py` script includes functionality to extract interactive elements from a webpage, execute actions like clicks and form submissions, and cache analysis results.

## Features

- Analyze visible and interactive elements on a webpage.
- Execute actions such as clicks, form submissions, or input field interactions.
- Cache analysis results to optimize repeated queries.
- Designed with FastAPI, Pydantic, and asyncio for robust and asynchronous operations.
- Uses `uvicorn` as the ASGI server for serving the FastAPI application.

---

## Requirements

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- Google Chrome (or Chromium browser)
- `uvicorn` for running the server
- Environment variable configuration with `python-dotenv`

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Duss02/server-gdg.git
   cd server-gdg
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Environment Variables

The server relies on environment variables for configuration. Create a `.env` file in your project directory and include the following variables:

```env
GEMINI_API_KEY=<Your_Google_Generative_AI_API_Key>
```

Replace `<Your_Google_Generative_AI_API_Key>` with your actual API key for accessing Google Generative AI.

---

## Usage

### Run the Server with Uvicorn

To start the server, use the following command:

```bash
uvicorn examples.server:app --host 0.0.0.0 --port 8000 --reload
```

This will start the server on `http://0.0.0.0:8000`. The `--reload` flag enables live reloading during development.

---

## API Endpoints

### 1. **Root Endpoint**

```http
GET /
```

Returns a simple "Hello, World" response to verify the server is running.

---

### 2. **Analyze a Webpage**

```http
POST /analyze-webpage
```

Analyze a webpage and list all the interactive elements.

**Request Body:**

```json
{
  "link": "https://example.com"
}
```

**Response:**

Returns a JSON object describing the interactive elements on the webpage.

---

### 3. **Execute an Action**

```http
POST /execute-action
```

Perform a specific action on the currently open webpage.

**Request Body:**

- Click example:

  ```json
  {
    "action_type": "click",
    "element_id": "button-id",
    "element_label": "Submit"
  }
  ```

- Input example:

  ```json
  {
    "action_type": "input",
    "element_id": "input-id",
    "element_label": "Search",
    "value": "FastAPI"
  }
  ```

**Response:**

Returns the result of the action, including any page changes or feedback.

---

### 4. **Current Page Information**

```http
GET /current-page
```

Retrieves information about the currently open page.

**Response:**

```json
{
  "success": true,
  "page_info": {
    "url": "https://example.com"
  }
}
```

---

## Browser Configuration

The server uses a browser automation tool to interact with web pages. Ensure you have Google Chrome installed and accessible. The `server.py` script uses the following browser configuration:

- **Browser Path:** `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
- **Headless Mode:** Disabled
- **Custom Profile Directory:** `remote-debug-profile`

If necessary, update the browser path in the `server.py` script under the `BrowserConfig` section.

---

## Development

### Running the Server

To run the script and the server, execute:

```bash
python examples/server.py
```

This will start the FastAPI server.

### Debugging

Ensure that `logging` is set to `INFO` level to capture detailed logs:

```python
logging.basicConfig(level=logging.INFO)
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to customize this README further based on your project needs!
