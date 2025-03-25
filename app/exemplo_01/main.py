import requests

from rich.console import Console

from app.settings import get_settings


settings = get_settings()

URL = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
    "Content-Type": "application/json", 
}

data = {
    "model": "gpt-4-turbo",
    "messages": [
        {"role": "system", "content": "Você é um assistente útil."},
        {
            "role": "user",
            "content": "Me explique o que é IA generativa de forma simples.",
        },
    ],
    "temperature": 0.7,
}

response = requests.post(URL, headers=headers, json=data)

if response.status_code == 200:
    response_json = response.json()
    Console().print(response_json["choices"][0]["message"]["content"], style="bold green")
else:
    print("Erro:", response.status_code, response.text)
