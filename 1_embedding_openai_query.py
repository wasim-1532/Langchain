# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from torch import embedding


# load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# result=embeddings.embed_query("Delhi is the capital of India")
# print(str(result))

# from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# result = embeddings.embed_query("Delhi is the capital of India")
# print(result)


from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

result = embeddings.embed_query("Delhi is the capital of India")

# Keep exactly 32 dimensions
result = result[:32]

print(result)
print("Dimensions:", len(result))