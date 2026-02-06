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
def initialize_low_temp_high_tokens_model() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with low temperature and high token limit.
    
    This demonstrates the low temperature + high tokens setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI with temperature=0.2, max_output_tokens=100
    3. The .invoke() method will be called with messages in the API endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
    # ChatOpenAI with low temperature and high token limit
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,  # Low temperature for focused responses
        max_tokens=100,  # High token limit for detailed responses
        max_retries=2,
    )


def initialize_high_temp_high_tokens_model() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with high temperature and high token limit.
    
    This demonstrates the high temperature + high tokens setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI with temperature=1.0, max_output_tokens=100
    3. The .invoke() method will be called with messages in the API endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
    # ChatOpenAI with high temperature and high token limit
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=1.0,  # High temperature for creative responses
        max_tokens=150,  # High token limit for detailed responses
        max_retries=2,
    )


def initialize_medium_temp_low_tokens_model() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client with medium temperature and low token limit.
    
    This demonstrates the medium temperature + low tokens setup:
    1. Environment Setup: Load the GEMINI_API_KEY from a .env file
    2. Instantiation: Create an instance of ChatOpenAI with temperature=0.5, max_output_tokens=15
    3. The .invoke() method will be called with messages in the API endpoint

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
    """
       # ChatOpenAI with medium temperature and low token limit
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.5,  # Medium temperature for balanced responses
        max_tokens=50,  # Low token limit for truncated responses
        max_retries=2,
    )


def generate_response(prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Generate a response using the Gemini model with specified parameters.
    
    Args:
        prompt: The input prompt
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
        
    Returns:
        The generated response content
    """
    try:
        # ChatOpenAI with custom parameters
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=2,
        )
        
        result = llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")


app = FastAPI(title="Practice LO2: Temperature and Token Experimentation", version="1.0.0")

# Initialize all three models
low_temp_high_tokens_llm = initialize_low_temp_high_tokens_model()
high_temp_high_tokens_llm = initialize_high_temp_high_tokens_model()
medium_temp_low_tokens_llm = initialize_medium_temp_low_tokens_model()


class ChatRequest(BaseModel):
    message: str


class ExperimentResponse(BaseModel):
    low_temp_high_tokens_response: str
    high_temp_high_tokens_response: str
    medium_temp_low_tokens_response: str


@app.post("/experiment", response_model=ExperimentResponse)
def experiment(request: ChatRequest) -> ExperimentResponse:
    """
    Temperature and token experimentation endpoint demonstrating parameter effects.
    
    This demonstrates the experimentation workflow:
    1. Environment Setup: GEMINI_API_KEY loaded from .env file
    2. Triple Instantiation: Create three models with different temperature and token settings
    3. Parallel Invocation: Call all three models with the same prompt
    4. Response Comparison: Return all three responses for analysis
    5. Output: Observe how temperature affects creativity and max_tokens affects response length
    """
    try:
        creative_prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Step 3: Parallel Invocation - Call all three models with the same prompt
        response_1 = low_temp_high_tokens_llm.invoke(creative_prompt)
        response_2 = high_temp_high_tokens_llm.invoke(creative_prompt)
        response_3 = medium_temp_low_tokens_llm.invoke(creative_prompt)
        
        # Step 4 & 5: Response Handling and Output
        content_1 = response_1.content if hasattr(response_1, "content") else str(response_1)
        content_2 = response_2.content if hasattr(response_2, "content") else str(response_2)
        content_3 = response_3.content if hasattr(response_3, "content") else str(response_3)
        
        return ExperimentResponse(
            low_temp_high_tokens_response=content_1,
            high_temp_high_tokens_response=content_2,
            medium_temp_low_tokens_response=content_3
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running experiments: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
