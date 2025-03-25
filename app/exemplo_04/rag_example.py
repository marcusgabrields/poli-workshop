import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

from app.settings import get_settings

settings = get_settings()

# 1. Configuração das chaves de API
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_NAMESPACE = "workshop"
OPENAI_API_KEY = settings.OPENAI_API_KEY

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "workshop"


# Create a Pinecone client instance
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
INDEX_NAME = "workshop-index"
pinecone_client = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=PINECONE_NAMESPACE,
)


# Inicializa a conexão com o Pinecone

# 2. Carrega o PDF
loader = PyPDFLoader("app/exemplo_03/documento.pdf")  # Substitua pelo seu arquivo
documents = loader.load()

# 3. Divide o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


vectorstore = pinecone_client.from_documents(docs, embeddings, index_name=INDEX_NAME)

custom_prompt = "Responda com o sotaque de um Recifense."

# 5. Cria o chatbot com RAG
template = """
Seu nome é Amdir, você é um assistente especialista em "O senhor dos anéis", sua função é responder a perguntas sobre o assunto.
Responda a pergunta  baseado no contexto fornecido.

%s.

Contexto: {context}

Pergunta: {question}

Answer:
"""
prompt = PromptTemplate(
    template=template % (custom_prompt),
    input_variables=[
        "context",
        "question",
    ],
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}, search_type="similarity", score_threshold=0.9
)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=OPENAI_API_KEY),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)

# 6. Faz perguntas ao chatbot
while True:
    pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        break

    resposta = qa_chain.run(pergunta)
    print("\n🤖 Resposta:", resposta)
