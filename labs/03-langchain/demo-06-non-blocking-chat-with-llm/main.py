import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv()


def initialize_gemini_llm() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client for async operations.
    
    This demonstrates the async LLM setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI
    3. The .ainvoke() method will be called with messages in the async endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    model_name = os.getenv("GEMINI_MODEL_NAME")
    if not model_name:
        raise ValueError("GEMINI_MODEL_NAME is not set. Please configure it in your .env file.")
    if not base_url:
        raise ValueError("GEMINI_BASE_URL is not set. Please configure it in your .env file.")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it in your .env file.")

    # ChatOpenAI for async operations
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
    )


async def async_generate_response(prompt: str) -> str:
    """
    Define Coroutine: Create a function using async def.
    
    This demonstrates the async workflow:
    1. Define Coroutine: Create a function using async def
    2. Instantiate LLM: Create the ChatOpenAI instance as usual
    3. Await ainvoke: Inside the coroutine, call the .ainvoke() method and use the await keyword
    4. Run with asyncio: Use asyncio.run() to execute the top-level coroutine
    
    Args:
        prompt: The input prompt
        
    Returns:
        The generated response content
    """
    try:
        # Step 2: Instantiate LLM
        llm = initialize_gemini_llm()
        
        # Step 3: Await ainvoke - Inside the coroutine, call the .ainvoke() method and use the await keyword
        result = await llm.ainvoke(prompt)
        
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raise Exception(f"Error generating async response: {str(e)}")


app = FastAPI(title="Single Async Call Demo", version="1.0.0")


class ChatRequest(BaseModel):
    message: str


class AsyncResponse(BaseModel):
    response: str
    model: str
    execution_type: str


@app.post("/async-chat", response_model=AsyncResponse)
def async_chat(request: ChatRequest) -> AsyncResponse:
    """
    Single async call endpoint demonstrating the async workflow:
    
    1. Define Coroutine: Create a function using async def
    2. Instantiate LLM: Create the ChatChatOpenAI instance as usual
    3. Await ainvoke: Inside the coroutine, call the .ainvoke() method and use the await keyword
    4. Run with asyncio: Use asyncio.run() to execute the top-level coroutine
    """
    try:
        prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Step 4: Run with asyncio - Use asyncio.run() to execute the top-level coroutine
        response = asyncio.run(async_generate_response(prompt))
        
        return AsyncResponse(
            response=response,
            model=os.getenv("GEMINI_MODEL_NAME"),
            execution_type="async"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    

    uvicorn.run(app, host="0.0.0.0", port=8000)
