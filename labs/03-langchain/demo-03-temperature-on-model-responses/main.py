import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from langchain_openai import ChatOpenAI



# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL_NAME")
base_url = os.getenv("GEMINI_BASE_URL")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please configure it in your .env file.")
if not model_name:
    raise ValueError("GEMINI_MODEL_NAME is not set. Please configure it in your .env file.")
if not base_url:
    raise ValueError("GEMINI_BASE_URL is not set. Please configure it in your .env file.")
def initialize_factual_model() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with low temperature for factual responses.
    
    This demonstrates the factual model setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI with temperature=0.1
    3. The .invoke() method will be called with messages in the API endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
   

    # ChatOpenAI with low temperature for factual responses
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,  # Low temperature for consistent, factual responses
        max_retries=2,
    )


def initialize_creative_model() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with high temperature for creative responses.
    
    This demonstrates the creative model setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI with temperature=0.9
    3. The .invoke() method will be called with messages in the API endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
    # ChatOpenAI with high temperature for creative responses
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.9,  # High temperature for creative, varied responses
        max_retries=2,
    )


app = FastAPI(title="Temperature Experiment: LLM Output Comparison", version="1.0.0")

# Initialize both models
factual_llm = initialize_factual_model()
creative_llm = initialize_creative_model()
model_name=model_name


class ChatRequest(BaseModel):
    message: str


class TemperatureResponse(BaseModel):
    factual_response: str
    creative_response: str
    factual_temperature: float
    creative_temperature: float
    model: str




@app.post("/compare-temperatures", response_model=TemperatureResponse)
def compare_temperatures(request: ChatRequest) -> TemperatureResponse:
    """
    Temperature comparison endpoint demonstrating how different temperature values affect LLM outputs.
    
    This demonstrates the temperature experiment workflow:
    1. Environment Setup: GEMINI_API_KEY loaded from .env file
    2. Dual Instantiation: Create both factual (temp=0.1) and creative (temp=0.9) models
    3. Parallel Invocation: Call .invoke() on both models with the same prompt
    4. Response Comparison: Return both responses for side-by-side analysis
    5. Output: Observe how low-temperature response is more conventional while high-temperature is more creative
    """
    try:
        prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Step 3: Parallel Invocation - Call both models with the same prompt
        factual_result = factual_llm.invoke(prompt)
        creative_result = creative_llm.invoke(prompt)
        
        # Step 4 & 5: Response Handling and Output
        factual_content = factual_result.content if hasattr(factual_result, "content") else str(factual_result)
        creative_content = creative_result.content if hasattr(creative_result, "content") else str(creative_result)
        
        return TemperatureResponse(
            factual_response=factual_content,
            creative_response=creative_content,
            factual_temperature=0.1,
            creative_temperature=0.9,
            model=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
