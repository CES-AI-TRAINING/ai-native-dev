# LLM Streaming Endpoint with Server-Sent Events (SSE)

This project demonstrates how to implement **streaming AI responses** using **FastAPI** and **Server-Sent Events (SSE)**.
It shows how to create a real-time, word-by-word streaming experience similar to ChatGPT's typing effect.
It is implemented using **uv** for environment management.

---

## Features

- **Streaming Endpoint:** `/query/stream` for real-time token-by-token responses
- **Server-Sent Events (SSE):** Standard protocol for server-to-client streaming
- **Better User Experience:** Users see responses immediately as they're generated
- **CORS Support:** Enable browser-based clients
- **Error Handling:** Graceful streaming error messages
- **OpenAI SDK Integration:** Works with Gemini via OpenAI-compatible interface
- **Auto-Generated Documentation:** Interactive Swagger UI at `/docs`

---

## Project Structure

```bash
demo-08-llm-stram-endpoint/
├── main.py                      # FastAPI application with streaming
├── .env                        # Environment variables
├── .gitignore                  # Ignore sensitive/config files
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Lock file
└── README.md                   # Documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-08-llm-stram-endpoint
cd demo-08-llm-stram-endpoint
```

---

### 2. Create Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
uv add fastapi uvicorn openai python-dotenv
```

---

### 4. Configure Environment Variables

Change the filename from `.envbackup` to `.env`, and include the appropriate keys within it.

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL_NAME=gemini-2.5-flash
```

## **Note**: The GEMINI_MODEL value can be updated to any supported model. Model names may change over time, so always refer to the latest options in Google’s documentation.

## Usage

Run the script using `uv`:

```bash
uv run uvicorn main:app --reload --port 8000
```

The server will start at: http://localhost:8000

---

## Example Output

### Request:

```json
POST /query/stream
{
  "prompt": "What is AI?"
}
```

### Response (SSE):

```
data: Artificial

data:  Intelligence

data:  (

data: AI

data: )

data:  is

data:  the

data:  simulation

data:  of

data:  human

data:  intelligence

data:  processes

data:  by

data:  machines

data: .


```

---

## Key Concepts

| Step | Concept               | Description                      |
| ---- | --------------------- | -------------------------------- |
| 1    | Initialize Client     | Same as non-streaming setup      |
| 2    | Enable Streaming      | Add `stream=True` to API call    |
| 3    | Receive Stream        | Response returned as a generator |
| 4    | Iterate Chunks        | Loop through each streamed token |
| 5    | Display Progressively | Send chunks using SSE format     |

---

## Summary

This project demonstrates how to implement **streaming AI responses** using **FastAPI**, **Server-Sent Events (SSE)**, and **Gemini API**.
It improves **user experience** by showing results in real time, creating a **ChatGPT-style typing effect** while maintaining the same total processing time.
Perfect for building **real-time AI applications** and **interactive chat interfaces**.

curl -X POST http://127.0.0.1:8000/query/stream \
 -H "Content-Type: application/json" \
 -d '{"prompt": "What is the generator in python?"}'
