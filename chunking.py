from langchain_text_splitters import CharacterTextSplitter

class Chunking():
    @staticmethod
    def chunking(page_text, chunk_size = 900, chunk_overlap = 150):        
        all_chunks = []
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for page in page_text:
            page_text = page['text']
            page_no = page['page']

            chunks = splitter.split_text(page_text)

            for ch in chunks:
                all_chunks.append({
                    "text" : ch,
                    "page": page_no
                })
        
        return all_chunks

