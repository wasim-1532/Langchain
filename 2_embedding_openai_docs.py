from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

documents=[
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France."
]

result = embeddings.embed_documents(documents)

# Keep exactly 32 dimensions
result = result[:32]

print(result)
print("Dimensions:", len(result))