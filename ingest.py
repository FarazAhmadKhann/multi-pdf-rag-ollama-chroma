from config import  COLLECTION_NAME
from config import EMBEDDING_MODEL
import ollama
import chromadb
from pypdf import PdfReader
from config import CHUNK_OVERLAP
from config import CHUNK_SIZE
import hashlib
from config import DOC_DIR
pdf_path= list(DOC_DIR.glob('*.pdf'))
#pdf hashing
def calculate_pdf_hash(pdf_file):
    sha256 = hashlib.sha256()
    with open(pdf_file,"rb") as pdf_file:
        while True:
            pdf_block=pdf_file.read(1024*1024)
            if not pdf_block:
               break

            sha256.update(pdf_block)
    return sha256.hexdigest()
#dividing pdf into chunks
def chunk_extraction(pdf_file):
    pdf_final=[]
    pdf_content=[
        (pdf_page,(chunks.extract_text() or"").split())
         for pdf_page,chunks in enumerate(pdf_file.pages,start=1)
    ]
    for i in range(len(pdf_content)):
        for j in range(0,len(pdf_content[i][1]),CHUNK_SIZE-CHUNK_OVERLAP):
                pdf_chunk=pdf_content[i][1][j:j+CHUNK_SIZE]
                pdf_final.append((pdf_content[i][0],pdf_chunk))

    pdf_sentence = [
        (page, " ".join(text))
        for page, text in pdf_final
    ]

    return pdf_sentence
#EMBEDDING_CHUNKS
def embedding_chunks(pdf_sentence):
    final_pdf_embed=[]
    chunked_sentence=[
        text.strip()
        for page, text in pdf_sentence
    ]
    embedding_sentence=ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunked_sentence
    )

    for i in range(len(chunked_sentence)):
        page_no=pdf_sentence[i][0]
        original_chunk=chunked_sentence[i]
        embedding_chunk=embedding_sentence["embeddings"][i]
        final_pdf_embed.append((page_no,original_chunk,embedding_chunk))

    return final_pdf_embed
#VECTOR_DATABASE:
def store_chunks_in_chroma(collection,final_pdf_embed,sha_pdf,pdf_name):

    collection.upsert(
        ids=[f"Pdf name:{pdf_name} CHUNK  NO : {i}"
             for i in range(len(final_pdf_embed))
             ],
        embeddings=[
            embedding[2]
            for embedding in final_pdf_embed
        ],
        documents=[
            document[1]
            for document in final_pdf_embed],
        metadatas=[{
            "pdf_name": pdf_name,
            "pdf_hash": sha_pdf,
            "page_no":pdf[0],
            "chunk_no":i
        }
        for i,pdf in enumerate(final_pdf_embed)
        ]
    )
def existing_record(collection,pdf_hash):
    record=collection.get(
        where={
                "pdf_hash":pdf_hash
        },
    limit=1
   )
    return bool(record['ids'])

#main
def main():
    database = chromadb.PersistentClient("./Chroma_db")
    collection = database.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"}
    )
    if not pdf_path:
        print("No pdf file provided")
        return
    else:
        for pdf_file in pdf_path:
            responses=PdfReader(pdf_file)
            pdf_hash=calculate_pdf_hash(pdf_file)
            if existing_record(collection,pdf_hash):
                continue
            pdf_chunk=chunk_extraction(responses)
            if not pdf_chunk:
                continue
            pdf_embeds=embedding_chunks(pdf_chunk)
            store_chunks_in_chroma(collection,pdf_embeds,pdf_hash,pdf_file.name)
if __name__ == "__main__":
     main()