from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rich.console import Console


from app.settings import get_settings

settings = get_settings()



model = OpenAIModel(
    "gpt-4o",
    provider=OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
    ),
)
agent = Agent(model, system_prompt="Você é um assistente útil.")
 
result = agent.run_sync('Me explique o que é IA generativa de forma simples.')

Console().print(result.data, style="bold green")

