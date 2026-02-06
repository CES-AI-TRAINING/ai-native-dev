import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()


def initialize_gemini_llm_with_retry() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with retry mechanism.
    
    This demonstrates the enhanced LLM invocation workflow with retry:
    1. Environment Setup: Load the GEMINI_API_KEY from .env file
    2. Instantiation with Retries: Create ChatOpenAI with max_retries parameter
    3. The .invoke() method will automatically handle retries on failures

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it in your .env file.")
    if not model_name:
        raise ValueError("GEMINI_MODEL_NAME is not set. Please configure it in your .env file.")
    if not base_url:
        raise ValueError("GEMINI_BASE_URL is not set. Please configure it in your .env file.")
    # ChatOpenAI with retry mechanism
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,  # Retry up to 2 times on failure
    )


app = FastAPI(title="LangChain + Gemini with Retry Demo", version="1.0.0")
llm = initialize_gemini_llm_with_retry()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Basic LLM invocation endpoint with retry mechanism demonstrating the enhanced workflow:
    
    1. Environment Setup: GEMINI_API_KEY loaded from .env file
    2. Instantiation with Retries: ChatOpenAI created with max_retries=2
    3. Invocation: Call the .invoke() method with system prompt and user message
    4. Error Handling (Internal): LangChain automatically retries on retriable errors (HTTP 500, etc.)
    5. Output: Return the final AIMessage content after successful retry attempts
    """
    try:
        # Step 3: Invocation - Call the .invoke() method with system prompt and message
        system_prompt = "You are a helpful assistant. Please response to the user's message."
        full_prompt = f"{system_prompt}\n\n {request.message}\n"
        result = llm.invoke(full_prompt)
        
        # Step 4 & 5: Response Handling and Output - Extract content from AIMessage
        content = result.content if hasattr(result, "content") else str(result)
        model_name = os.getenv("GEMINI_MODEL_NAME")
        return ChatResponse(response=content, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
