import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

load_dotenv()

token = os.getenv("OPENROUTER_API_KEY")
if not token:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Create an OpenRouter API key and set "
        "it in your PowerShell session before running this script."
    )

agent = CodeAgent(
    model=OpenAIModel(
        model_id="openrouter/free",
        api_base="https://openrouter.ai/api/v1",
        api_key=token,
    ),
    tools=[],
)

agent.run("Calculate the average of [1, 2, 3, 4, 5] and return the result.")