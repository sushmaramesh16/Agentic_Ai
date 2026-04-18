import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

FINANCIAL_DOCS = [
    # Credit Score Rules
    "Credit score 800-850: Exceptional credit. Eligible for all loan types with best interest rates.",
    "Credit score 740-799: Very good credit. Approved for most loans with competitive rates.",
    "Credit score 670-739: Good credit. Generally approved with standard interest rates.",
    "Credit score 650-669: Fair credit. Manual review required for loans above 50k USD.",
    "Credit score 580-649: Poor credit. Only eligible for secured loans below 25k USD.",
    "Credit score below 580: Very poor credit. Application rejected for all unsecured loans.",
    # Loan-to-Income Rules
    "Income requirement: Minimum annual income must be at least 2.5x the loan amount requested.",
    "Debt-to-income ratio: Total monthly debt payments cannot exceed 43% of gross monthly income.",
    "High income earners above 200k annually may qualify for premium loan products up to 2 million USD.",
    # Loan Amount Rules
    "Loans above 75k USD require a minimum credit score of 700.",
    "Loans above 200k USD require minimum credit score of 750 and collateral documentation.",
    "Loans above 500k USD require credit score 800 or above and full asset verification.",
    "Personal loans are capped at 50k USD regardless of credit score.",
    # Employment Rules
    "Employment requirement: Applicant must be continuously employed for at least 1 year.",
    "Self-employed applicants must provide 2 years of tax returns and show stable income.",
    "Unemployed applicants are rejected for all unsecured loan products.",
    # Special Circumstances
    "Bankruptcy in last 7 years: Automatic rejection regardless of current credit score.",
    "Bankruptcy discharged more than 7 years ago: Eligible with credit score above 650.",
    "First-time borrowers with no credit history: Eligible for secured loans up to 10k USD only.",
    "Co-signer with credit score above 750 can improve approval chances for borderline applicants.",
    # Green Scenarios
    "Applicant with credit score above 750, income 3x loan amount, employed 2+ years: Automatic GREEN approval.",
    "Applicant with perfect payment history and credit score above 800: Priority approval with best rates.",
    # Interest Rate Guidelines
    "Interest rate for GREEN applicants: 4-7% APR depending on loan term.",
    "Interest rate for YELLOW applicants: 8-12% APR with additional conditions.",
    "Interest rate for borderline applicants: 13-18% APR for secured loans only.",
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
