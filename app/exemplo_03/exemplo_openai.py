import pypdf
from openai import OpenAI
from rich.console import Console

from app.settings import get_settings

settings = get_settings()


def extrair_text_do_pdf(pdf_path):
    """Lê um PDF e extrai o texto."""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        text = "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )
    return text


def pergunta_para_o_gpt(context, question):
    """Faz uma pergunta para o GPT baseado no contexto fornecido."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    system_prompt = (
        "Você é um assistente que responde as perguntas apenas com base no contexto fornecido."
    )
    pergunta_com_contexto = f"Contexto: {context}\n\nPergunta: {question}"

    Console().print("Prompt:", style="bold red")
    Console().print(system_prompt, style="bold red")
    Console().print(pergunta_com_contexto, style="bold red")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": pergunta_com_contexto},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# Caminho do PDF (altere para o caminho correto do seu arquivo)
pdf_path = "app/exemplo_03/documento.pdf"

# Extrair texto do PDF
context = extrair_text_do_pdf(pdf_path)

# Fazer uma pergunta baseada no conteúdo do PDF
pergunta = "Qual é o tema principal do documento?"

resposta = pergunta_para_o_gpt(context, pergunta)

print()
print()

Console().print(resposta, style="bold green")
