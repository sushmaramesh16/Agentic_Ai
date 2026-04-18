import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

FINANCIAL_DOCS = [
    "Credit score below 650: applicant is rejected for any loan above 50k USD.",
    "Credit score between 650 and 699: yellow flag, manual review required before approval.",
    "Credit score 700 and above: green light, applicant is eligible for loan approval.",
    "Income requirement: minimum annual income must be at least 2.5 times the loan amount.",
    "Debt-to-income ratio: total monthly debt payments cannot exceed 43% of gross monthly income.",
    "Loans above 75k USD require a minimum credit score of 700.",
    "First-time borrowers with no credit history are automatically flagged as high risk.",
    "Applicants with recent bankruptcy in last 7 years are rejected regardless of credit score.",
    "Employment requirement: applicant must be employed for at least 1 year continuously.",
    "Loan amounts above 200k require additional collateral documentation.",
]

def ingest_documents():
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/chroma_db")

    try:
        client.delete_collection("financial_guidelines")
    except:
        pass

    collection = client.create_collection(
        name="financial_guidelines",
        embedding_function=embedding_fn
    )

    collection.add(
        documents=FINANCIAL_DOCS,
        ids=[f"doc_{i}" for i in range(len(FINANCIAL_DOCS))]
    )

    print(f"✅ Ingested {len(FINANCIAL_DOCS)} documents into ChromaDB!")

if __name__ == "__main__":
    ingest_documents()