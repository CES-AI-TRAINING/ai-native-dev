# Sentiment Analysis LCEL Chain Demo

A FastAPI application that demonstrates a multi-step LCEL chain with custom logic using RunnableLambda.
This example analyzes customer feedback, classifies sentiment, determines urgency, and generates a response strategy.

## Features

- **Multi-Step LCEL Workflow**: Sequential processing across seven steps
- **Custom Logic with RunnableLambda**: Adds domain-specific transformations
- **Automatic Data Passing**: No need to manually handle intermediate outputs
- **FastAPI Endpoint**: Exposes the analysis as a REST API

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux:**

   ```bash
   cd demo-15-sentiment-analysis-lcel-chain-demo
   ```

   **For Windows:**

   ```cmd
   cd demo-15-sentiment-analysis-lcel-chain-demo
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
   GEMINI_API_KEY=your_gemini_api_key_here
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

The app will start at `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## API Endpoints

### POST /analyze-feedback

Analyzes a customer's feedback, classifies sentiment, determines urgency, and provides a response strategy.

**Request Example:**
customer_feedback= I've been waiting for 3 weeks and still no response!

**Response Example:**

```json
{
  "feedback": "I've been waiting for 3 weeks and still no response!",
  "strategy": "Immediately apologize for the delay and escalate to a senior team member..."
}
```

## Testing the API

Visit http://localhost:8000/docs for interactive testing through Swagger UI.
