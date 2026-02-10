# Customer Support Assistant

A FastAPI application that uses Google's Gemini AI to generate customer support responses based on customer emails and issue categories.

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux:**

   ```bash
   cd demo-11-automating-customer-support-responses
   ```

   **For Windows:**

   ```cmd
   cd demo-11-automating-customer-support-responses
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

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## API Usage

### POST /generate-reply

Generate a customer support response based on the provided information.

**Request Body:**

```json
{
  "customer_email": "I received a damaged phone and need a replacement",
  "issue_category": "Product Replacement"
}
```

**Response:**

```json
{
  "reply": "I'm really sorry to hear about the damaged phone. Please provide your order number so we can arrange a replacement immediately."
}
```

## Features

- **Few-shot Examples**: Includes example interactions to guide the AI responses
- **Issue Categorization**: Tailors responses based on the type of customer issue
- **Professional Tone**: Generates empathetic and clear customer support responses
- **Stateless Processing**: Each request is processed independently without conversation memory
- **Simple API**: Clean interface requiring only customer email and issue category

## Error Handling

If you encounter authentication errors, ensure:

1. Your API key is correctly set in the `.env` file
2. The API key is valid and has the necessary permissions
3. You have an active internet connection
