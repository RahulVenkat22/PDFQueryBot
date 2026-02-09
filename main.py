

from chunking import Chunking
from llm_client import LLMClient
from pdf_reader import PDFReader
from vector_space import VectorStore


class Main_function():
    @staticmethod
    def main():
        # user_input = input("Enter the path location of pdf file")
        user_input = "Computer-Basics--computer_basics2.pdf"
        pages = PDFReader.load_pages(user_input)

        print("Loading progess was completed and chunking progress was intialized")

        chunking = Chunking.chunking(pages)

        store = VectorStore(persist_dir="./chroma_db") # Step 4: Add chunks to ChromaDB 
        store.add_chunks(chunking)

        print("Chunks added to vector store!")

        while True:
            question = input("\nAsk a question (or exit): ")
            if question.lower() == "exit" or question.lower() == "quit":
                break

            results = store.search(query = question)

            print("\nTop retrieved chunks:")
    
              # ✅ Send to Ollama and get final answer
            answer = LLMClient.ask_llm(question, results)

            print("\nFinal Answer (from PDF only):\n", answer)

Main_function.main()