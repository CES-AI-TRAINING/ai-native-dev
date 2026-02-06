import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    """Initialize and return a ChatOpenAI (Gemini) model instance."""
    # Get API details from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your .env file.")
    
    if not base_url:
        raise ValueError("GEMINI_BASE_URL environment variable is required. Please set it in your .env file.")
    
    if not model_name:
        raise ValueError("GEMINI_MODEL_NAME environment variable is required. Please set it in your .env file.")
    
    # Initialize and return ChatOpenAI instance
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key
    )

# Create FastAPI app
app = FastAPI(title="LangChain + Gemini Demo", version="1.0.0")
# Initialize model
llm = initialize_llm()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Basic LLM invocation endpoint demonstrating the fundamental workflow:
    
    1. Environment Setup: GEMINI_API_KEY, GEMINI_MODEL_NAME, and GEMINI_BASE_URL loaded from .env file
    2. Instantiation: ChatOpenAI instance created with model from GEMINI_MODEL_NAME environment variable
    3. Invocation: Call the .invoke() method with the user's message
    4. Response Handling: LangChain sends request to Gemini API and parses JSON response into AIMessage
    5. Output: Return the content of the AIMessage object
    """
    try:
        prompt = "You are a helpful assistant. Please response to the user's message. "
        full_prompt = f"{prompt}\n\n {request.message}\n"
        # Step 3: Invocation - Call the .invoke() method on the instance with messages
        result = llm.invoke(full_prompt)
        # Step 4: Response Handling - LangChain parses the JSON response into AIMessage
        # Step 5: Output - Extract content from the AIMessage object
        content = result.content if hasattr(result, "content") else str(result)
        model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        return ChatResponse(response=content, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


