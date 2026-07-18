import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader

dir_loader = DirectoryLoader(
    "./Data/pdfs",
    glob= "**/*.pdf", ##pattern to match files
    loader_cls = PyMuPDFLoader, ##Loader class to use
    show_progress = False
)
pdf_documents = dir_loader.load()
print(pdf_documents)
