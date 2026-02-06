# Single Async Call Demo: LangChain + Gemini

This project demonstrates the basic syntax for making a single async call using LangChain with Google Gemini through a FastAPI web service.

## Objective

To demonstrate the async workflow that:
- Defines a coroutine using `async def`
- Instantiates the ChatGoogleGenerativeAI instance
- Uses `.ainvoke()` method with `await` keyword
- Executes the coroutine with asyncio

## Conceptual Flow

1. **Define Coroutine**: Create a function using `async def`
2. **Instantiate LLM**: Create the ChatOpenAI instance as usual
3. **Await ainvoke**: Inside the coroutine, call the `.ainvoke()` method and use the `await` keyword to pause until the result is returned
4. **Run with asyncio**: Use `asyncio.run()` to execute the top-level coroutine

## Project Structure

```
single_async_call/
├── .env                    # Environment variables (API key)
├── main.py                # FastAPI application
├── pyproject.toml         # Project dependencies
├── README.md              # This file
└── .python-version        # Python version specification
```

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux:**
   ```bash
   cd demo-6-non-blocking-chat-with-llm
   ```

   **For Windows:**
   ```cmd
   cd demo-6-non-blocking-chat-with-llm
   ```

2. Install dependencies using UV:

   **For Linux/Windows (Same command):**
   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project environment

## Configuration

1. Create a `.env` file in the project root:

   **For Linux:**
   ```bash
   touch .env
   ```

   **For Windows (PowerShell):**
   ```powershell
   New-Item -Path .env -ItemType File
   ```

   **For Windows (CMD):**
   ```cmd
   type nul > .env
   ```

2. Add your Google Gemini API key to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   GEMINI_MODEL_NAME=gemini-2.5-flash
   ```
   
   **Note**: 
   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - The GEMINI_MODEL_NAME value can be updated to any supported model. Model names may change over time, so always refer to the latest options in Google’s documentation.

## Running the Application

**For Linux/Windows (Same commands):**
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## Testing the API

- Open your browser to `http://localhost:8000/docs` for interactive API documentation
- Send POST requests to `http://localhost:8000/async-chat`

## Features

- ✅ Environment variable loading with `python-dotenv`
- ✅ ChatOpenAI instantiation for "gemini-2.0-flash" model
- ✅ **Async coroutine definition** using `async def`
- ✅ **Async LLM invocation** using `.ainvoke()` with `await`
- ✅ FastAPI web service with automatic API documentation
- ✅ **Asyncio integration** for async operations
- ✅ Interactive API testing via Swagger UI

## API Endpoint

### POST /async-chat

Make a single async call to the Gemini model.

**Request Body:**
```json
{
  "message": "Your message here"
}
```

**Response:**
```json
{
  "response": "Generated response from async call...",
  "model": "gemini-2.0-flash",
  "execution_type": "async"
}
```

