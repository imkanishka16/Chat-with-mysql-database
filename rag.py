# import chromadb
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# load_dotenv()
# from langchain.chains import RetrievalQA

# # Load your ChromaDB collection
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
# chroma_collection = chroma_client.get_collection("tci_glossary")

# OPENAI_KEY = os.getenv('OPENAI_KEY')

# # Define the retriever function to get relevant chunks from Chroma
# def retrieve_documents(query, top_k=3):
#     results = chroma_collection.query(
#         query_texts=[query],
#         n_results=top_k
#     )
    
#     # Assuming 'documents' is a list of strings
#     retrieved_documents = results['documents'][0]
#     return retrieved_documents

# # Use GPT-4 as the LLM model through OpenAI's API
# llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4-0125-preview")

# # # Create a simple function for RAG
# # def rag_pipeline(question):
# #     # Retrieve documents based on the query
# #     retrieved_docs = retrieve_documents(question)
    
# #     # Combine retrieved docs into a single context for the LLM
# #     context = "\n".join(retrieved_docs)

# #     # Generate response using GPT-4 with the context
# #     response = llm(f"Context: {context}\n\nQuestion: {question}")
    
# #     return response



# def rag_response(question):
#     template = """
#     You are an assistant for question-answering tasks.
#     Answer the question based on the following context. If you don't know the answer, just say that you don't know.:
#     {context}
#     Question: {question}
#     Context: {context}
#     Answer:

#     Your answer should be in JSON format:
#     {
#         text answer: your answer
#     }

#     For Example:
#     Question:What is the meaning of Fund Name?
#     Answer:
#     {
#         text answer: Refers to the name of the donor fund providing public finance.
#     }
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=retrieve_documents(question),
#         chain_type_kwargs={"prompt": prompt}
#     )

#     result = qa_chain({"query": question })
#     return result


# question = "What is private finance?"
# response = rag_response(question)
# print(response)



import chromadb
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr
from typing import List

load_dotenv()

# Load your ChromaDB collection
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.get_collection("tci_glossary")

OPENAI_KEY = os.getenv('OPENAI_KEY')

class ChromaDBRetriever(BaseRetriever, BaseModel):
    """Custom retriever for ChromaDB that properly implements Pydantic BaseModel"""
    _collection: any = PrivateAttr()
    top_k: int = 3

    def __init__(self, **data):
        super().__init__(**data)
        self._collection = chroma_collection

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._collection.query(
            query_texts=[query],
            n_results=self.top_k
        )
        return [Document(page_content=doc) for doc in results['documents'][0]]

# Initialize the retriever
retriever = ChromaDBRetriever()


llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4o")

def rag_response(question: str) -> dict:
    
    template = """You are an assistant for question-answering tasks. 
    Use the following context to answer the question. If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}

    Provide a clear and direct answer without any JSON formatting or special characters.
    """

    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": prompt,
        }
    )

    try:
        raw_result = qa_chain.invoke({"query": question})
        # Get just the answer text and wrap it in the desired format
        answer_text = raw_result.get('result', '').strip()
        return {"text_answer": answer_text}
    except Exception as e:
        print(f"Error during chain execution: {str(e)}")
        return {"text_answer": "An error occurred while processing your question."}

# if __name__ == "__main__":
#     question = "What is definition of plastic pollusion?"
#     response = rag_response(question)
#     print(response)

# question = "What is definition of plastic pollusion?"
# response = rag_response(question)
# print(response)

