import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Environment configuration
API_KEY = os.getenv("PINECONE_API_KEY")
ENV = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "us-east1-aws"
BASE_NAME = os.getenv("PINECONE_BASE_INDEX_NAME")  # e.g. "revola"
EMBED_MODEL = "text-embedding-3-large"

dim = 3072  # for OpenAI text-embedding-3-large


def init_index(company: str):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = f"{BASE_NAME}-{company.lower()}"
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return None  # Critical error
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    return index


init_index("cirrascale")
