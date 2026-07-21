import ollama
import chromadb
from config import COLLECTION_NAME, EMBEDDING_MODEL,TOP_K,GENERATION_MODEL

database=chromadb.PersistentClient(
    path="./Chroma_db"
)
collection = database.get_collection(
    name=COLLECTION_NAME
)
def embed_question(question):
    embed_response=ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )
    return embed_response["embeddings"]
def similarity(embedding_questions,collections):
    response=collections.query(
        query_embeddings=embedding_questions,
        n_results=TOP_K
    )
    return response["documents"][0]
def generative_model_response(similar_answers,question):
    contexts="\n\n".join(
        similar_answers
    )
    prompt=f"""
            
            You are a helpful assistant answering questions from retrieved PDF content.
            
            Rules:
            1. Answer only using the provided context.
            2. Do not use outside knowledge.
            3. If the answer is not present in the context, say:
               "I could not find this information in the provided documents."
            4. Do not invent facts.
            5. Give a clear and concise answer.
            6. Mention the source PDF and page number when available.
            
            Context:
            {contexts}
            
            Question:
            {question}
            
            Answer:
            """
    response=ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]
def main():
    question=input("Enter question: ").strip()
    if not question:
        print("enter the question again")
        return
    embedded_question=embed_question(question)
    similar_answers=similarity(embedded_question,collection)
    generative_model_responses=generative_model_response(similar_answers,question)
    print(generative_model_responses)
if __name__=="__main__":
    main()
