import fitz
from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
