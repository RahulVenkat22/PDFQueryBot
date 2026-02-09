import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.Client(
            chromadb.config.Settings(persist_directory=persist_dir)
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks"
        )
        self.embed_model = SentenceTransformer("intfloat/e5-base")

    def add_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"]} for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        embeddings = self.embed_model.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def search(self, query, k=1):
        q_emb = self.embed_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=k
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"]
            })

        return hits
