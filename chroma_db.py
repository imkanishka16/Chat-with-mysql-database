import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



reader = PdfReader("pdf/RTP_Glossary.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.HttpClient(host='3.108.173.11', port=8000)
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.create_collection("tci_glossary", embedding_function=embedding_function,metadata={"hnsw:space": "l2"})

ids = [str(i) for i in range(len(character_split_texts))]

chroma_collection.add(ids=ids, documents=character_split_texts)
chroma_collection.count()
print("Data insertion successfull!...")
