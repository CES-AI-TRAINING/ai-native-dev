from langchain.agents import create_agent
import os
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

def main():
    print("Hello from practice-04-langchain-agent!")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL_NAME")

# 2. Create an agent using the loaded model and system prompt
    agent = create_agent(
        model=model,
        system_prompt="You are a helpful assistant"
    )

    #llm = ChatOpenAI(model=model, temperature=0.9, openai_api_key=api_key)

# 3. Invoke the agent with a user query and print the response
    prompt = {"messages": [{"role": "user", "content": "what is the capital of France?"}]};
    response =agent.invoke(prompt)

# 4. Print the content of the response
    print(response)

if __name__ == "__main__":
    main()
