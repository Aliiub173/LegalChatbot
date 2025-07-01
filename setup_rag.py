from rag.utils import extract_text_from_pdf, chunk_text
from rag.embedder import build_index

pdf_text = extract_text_from_pdf("data/marry.pdf.pdf")
chunks = chunk_text(pdf_text)
build_index(chunks)
