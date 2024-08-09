from src.helper import load_pdf , text_split , download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

import time 
index_name = "medicalchatbot"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f'there is no index name {index_name}')
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    index = pc.Index(index_name)

    vectore_store = PineconeVectorStore(index=index, embedding=embedding)
    uuids = [str(uuid4()) for _ in range(len(text_chunks))]
    vectore_store.add_documents(documents=text_chunks, ids=uuids)

    retriever = vectore_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
else:
    index = pc.Index(index_name)
    print(f'the index name {index_name} is there ')
    vectore_store = PineconeVectorStore(index=index, embedding=embedding)
    retriever = vectore_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
        # Example query
    query_text = "What are the symptoms of a cold?"

    # Perform the similarity search
    results = retriever.get_relevant_documents(query_text)
    print(results)