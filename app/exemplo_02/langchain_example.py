from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from rich.console import Console

from app.settings import get_settings

settings = get_settings()

chat = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model="gpt-4") # type: ignore

def chatbot(input_text):
    response = chat([HumanMessage(content=input_text)])
    return response.content

pergunta = "Explique a teoria da relatividade de forma simples."
resposta = chatbot(pergunta)

Console().print(resposta, style="bold green")
