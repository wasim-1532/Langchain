# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from torch import embedding

# load_dotenv()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# document = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query="tell me about Virat Kohli."

# doc_embeddings=embedding.embed_documents(document)
# query_embedding=embedding.embed_query(query)

# print(cosine_similarity([query_embedding], doc_embeddings))


from langchain_huggingface import HuggingFaceEmbeddings
from numpy.ma import sort
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Who is current Prime Minister of india?"

doc_embeddings = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)

best_index=similarity_scores.argmax()
print(query)
print(document[best_index][:100])