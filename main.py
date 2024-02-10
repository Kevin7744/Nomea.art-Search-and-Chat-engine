import os
from dotenv import load_dotenv

load_dotenv()  

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector


CONNECTION_STRING = os.getenv("SUPABASE_URL")

# Instantiate OpenAI embeddings
embeddings = OpenAIEmbeddings()

def embed_text(text):
    return embeddings.embed(text)

db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="documents" ,
    embedding_function=embed_text
)

query = input("Enter your query: ")


query_embedding = embeddings.embed(query)

results = db.similarity_search_with_score(query_embedding)

for doc, score in results:
    print(f"Document content: {doc.pageContent}\nSimilarity score: {score:.4f}\n")
