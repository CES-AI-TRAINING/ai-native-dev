# Max Tokens Limit Demo: LangChain + Gemini

This project demonstrates how to use `max_output_tokens` to strictly limit the length of generated responses using LangChain with Google Gemini through a FastAPI web service.

## Objective

To demonstrate the token limiting functionality that:
- Instantiates the Gemini model with a strict token limit
- Invokes it with prompts that would normally generate long responses
- Shows how responses are truncated at the specified token limit

## Conceptual Flow

1. **Instantiate with Token Limit**: Create an LLM instance, setting `max_output_tokens` to a small number (e.g., 10)
2. **Invoke with Long Prompt**: Call `.invoke()` with a prompt that asks for a detailed explanation
3. **Observe Truncation**: Note that the response is cut off abruptly after approximately 10 tokens, demonstrating the hard limit imposed by the parameter

## Project Structure

```
max_tokens_limit/
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
   cd demo-4-max-token
   ```

   **For Windows:**
   ```cmd
   cd demo-4-max-token
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
   GEMINI_MODEL_NAME=gemini-2.5-flash-lite #(Model name are subject to change, please check Gemini portal before using it)
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
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
- Or send a POST request to `http://localhost:8000/chat` with JSON body:
  ```json
  {
    "message": "Explain the concept of machine learning in detail"
  }
  ```

## Features

- ✅ Environment variable loading with `python-dotenv`
- ✅ **Token limiting with `max_output_tokens=10`** for strict response truncation
- ✅ FastAPI web service with automatic API documentation
- ✅ Pydantic models for request/response validation
- ✅ Token limiting functionality
- ✅ Interactive API testing via Swagger UI

```

