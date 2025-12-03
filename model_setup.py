from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# Initialize LLM
def llm_model():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=200
    )
    return llm

# Initialize Embeddings
def embed_model():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-3-large", "text-embedding-ada-002"
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return embeddings


