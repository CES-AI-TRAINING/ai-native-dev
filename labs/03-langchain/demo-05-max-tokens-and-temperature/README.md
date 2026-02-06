# Practice LO2: Temperature and Token Experimentation

This project demonstrates how to experiment with both temperature and max_output_tokens parameters using LangChain with Google Gemini through a FastAPI web service.

## Objective

To create a script that allows for experimentation with both temperature and max_output_tokens parameters to understand their effects on AI model responses.

## Requirements Implementation

1. **Create practice_lo2.py**: FastAPI-based implementation
2. **Import necessary libraries**: LangChain, FastAPI, Pydantic, dotenv
3. **Define generate_response function**: Takes prompt, temperature, and max_tokens
4. **Three LLM initialization functions**: Each with different parameter combinations
5. **Main experimentation**: Three different parameter combinations in one endpoint

## Project Structure

```
practice_lo2/
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
   cd demo-5-max-tokens-and-temperature
   ```

   **For Windows:**
   ```cmd
   cd demo-5-max-tokens-and-temperature
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
   GEMINI_MODEL_NAME=gemini-2.5-flash-lite #(Model Subject to Change, please Check Gemini portal for latest version)
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
- Send POST requests to `http://localhost:8000/experiment`

## Features

- ✅ Environment variable loading with `python-dotenv`
- ✅ **Three separate LLM initialization functions** with different parameter combinations
- ✅ **Temperature experimentation** (0.0 to 1.0) for creativity control
- ✅ **Token limit experimentation** for response length control
- ✅ FastAPI web service with automatic API documentation
- ✅ **Single endpoint** showing all three experiments
- ✅ Interactive API testing via Swagger UI

## Three LLM Initialization Functions

### 1. `initialize_low_temp_high_tokens_model()`
- **Temperature**: 0.2 (focused, deterministic)
- **Max Tokens**: 100 (detailed response)
- **Purpose**: Produces focused, detailed, and consistent responses

### 2. `initialize_high_temp_high_tokens_model()`
- **Temperature**: 1.0 (creative, random)
- **Max Tokens**: 100 (detailed response)
- **Purpose**: Produces creative, varied, and unpredictable responses

### 3. `initialize_medium_temp_low_tokens_model()`
- **Temperature**: 0.5 (balanced)
- **Max Tokens**: 100 (very short response)
- **Purpose**: Produces truncated, concise responses that end abruptly



## Use Cases

This parameter experimentation is useful for:
- **Content Generation**: Finding the right balance of creativity vs consistency
- **Response Length Control**: Matching output to UI constraints
- **A/B Testing**: Comparing different parameter combinations
- **Model Tuning**: Optimizing parameters for specific use cases
