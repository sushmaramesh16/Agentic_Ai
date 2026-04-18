import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def retrieve_documents(query: str, n_results: int = 3) -> list[str]:
    """Search ChromaDB for documents relevant to the user query."""
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection(
        name="financial_guidelines",
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    docs = results["documents"][0]
    print(f"📚 Retrieved {len(docs)} relevant documents from ChromaDB")
    return docs