import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()


def initialize_gemini_llm_with_token_limit() -> ChatOpenAI:
 
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
        max_tokens=100 # Max tokens for the entire response (input + output
    )


app = FastAPI(title="Max Tokens Limit Demo", version="1.0.0")
llm = initialize_gemini_llm_with_token_limit()



class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str
    


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
 
    
    try:
        # Step 3: Invocation - Call the .invoke() method with system prompt and message
        system_prompt = "You are a helpful assistant. Please provide a detailed explanation."
        full_prompt = f"{system_prompt}\n\n {request.message}\n"
        result = llm.invoke(full_prompt)
        
        # Step 4 & 5: Response Handling and Output - Extract content from AIMessage
        content = result.content if hasattr(result, "content") else str(result)
        model_name = os.getenv("GEMINI_MODEL_NAME")
             
        return ChatResponse(
            response=content, 
            model=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
