"""
Demo 3: LLM Streaming Endpoint with Server-Sent Events (SSE)

This example demonstrates how to implement streaming AI responses
using FastAPI and Server-Sent Events for real-time token-by-token output.
"""

# 1. Import dependencies: FastAPI, CORS middleware, StreamingResponse, Pydantic, OpenAI, os, dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

# 2. Load environment variables from .env file api key, base url, and model name 
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

base_url = os.getenv("GEMINI_BASE_URL")
model = os.getenv("GEMINI_MODEL_NAME")

# 3. Initialize OpenAI client for Gemini API
client = OpenAI(api_key=api_key, base_url=base_url)

# 4. Initialize FastAPI application with metadata
app = FastAPI(
    title="AI Streaming API",
    description="Streaming LLM responses using Server-Sent Events",
    version="1.0.0"
)

# 5. Configure CORS middleware
# This allows browser-based clients from any origin ("*") to access the API
# WARNING: For production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Define Request Model
class QueryRequest(BaseModel):
    """Request model for user prompts"""
    prompt: str = "What is the list in python?"

# 7. Define Streaming Function
def stream_ai_response(user_prompt: str):
    """
    Stream AI response word-by-word using Server-Sent Events.
    
    Args:
        user_prompt: User's question or prompt
        
    Yields:
        str: SSE-formatted response chunks
    """
    try:
        # 8. Create streaming request to AI
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            stream=True
        )
        
        # 9. Yield each chunk as it arrives
        for chunk in stream:
            word = chunk.choices[0].delta.content or ""
            if word:
                yield f"data: {word}\n\n"
        
        # 10. Signal completion
        yield "\n\n"
        
    except Exception as error:
        yield f"data: [ERROR: {str(error)}]\n\n"

# 11. Define Streaming Endpoint
@app.post("/query/stream")
def ask_ai_streaming(request: QueryRequest):
    """
    Stream AI response word-by-word in real-time.
    
    Args:
        request: Request object containing user prompt
        
    Returns:
        StreamingResponse: SSE stream of AI response
    """
    return StreamingResponse(
        stream_ai_response(request.prompt),
        media_type="text/event-stream"
    )
