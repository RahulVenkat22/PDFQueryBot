from langchain_community.document_loaders import PDFPlumberLoader


class PDFReader():
    @staticmethod
    def load_pages(pdf_path: str):
        loader = PDFPlumberLoader(pdf_path)
        pages = []

        for doc in loader.lazy_load():
            text = doc.page_content
            page_no = doc.metadata.get("page", None)

            if text and text.strip():
                pages.append({
                    "page": page_no,
                    "text": text
                })

        return pages


