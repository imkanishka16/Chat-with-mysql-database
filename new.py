# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.utilities import SQLDatabase
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# import streamlit as st
# from sqlalchemy import create_engine
# import json
# import re
# from typing import Tuple, List,Dict,Any

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain_openai import OpenAIEmbeddings


# import os
# import pandas as pd

# load_dotenv()

# # Load OpenAI API key
# OPENAI_KEY = os.getenv('OPENAI_KEY')
# user = os.getenv('DB_USER')
# password = os.getenv('DB_PASSWORD') 
# host = os.getenv('DB_HOST')
# database = os.getenv('DB_NAME')


# # Create a SQLAlchemy engine
# #engine = create_engine(os.getenv('DB_CONN_STRING'))
# engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# # Wrap the engine with SQLDatabase
# db = SQLDatabase(engine)

# def _strip(text: str) -> str:
#     return text.strip()



# database_store = 'investment_tci_prompt'
# engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")
# from sqlalchemy import Column, Integer, Text, DateTime
# from sqlalchemy.orm import declarative_base, sessionmaker
# from datetime import datetime

# Base = declarative_base()

# class UserQuery(Base):
#     __tablename__ = 'user_queries'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     user_query = Column(Text, nullable=False)
#     timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

# # Create the table
# Base.metadata.create_all(engine2)

# def store_user_query(query: str, engine):
#     if not query:  # Check if query is None or an empty string
#         print("Error: Query cannot be empty.")
#         return
    
#     session = sessionmaker(bind=engine)()
#     new_query = UserQuery(user_query=query, timestamp=datetime.now())
#     session.add(new_query)
#     session.commit()
#     session.close()




# def get_sql_chain(db):
#   template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.This dataset consists of the global financial flows from both public and private sectors directed to tackle plastic pollution. The dataset covers multiple data points for each financial flow, including the time period, the name, institution type, and geography of both the flow provider and recipient, the application of the financial flow, and the flow amount based on multiple types of financial flow, such as loan, equity, or grant.
#     Based on the table schema below, write a SQL query that would answer the user's question.
    
#     <SCHEMA>{schema}</SCHEMA>
    
#     Conversation History: {chat_history}
    
#     Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
#     DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)to the database.
  
#     Instruction:
#     -To filter 'Domestic Finance', 'Development Finance','Private Finance' and 'MEA Finance', always use the 'sub_category' column.
#     -To filter 'Circular Pathways', 'Non-circular Pathways' and 'Supporting Pathways', always use the 'pathways' column.
#     -There is columns like ft_grant. don't get 'SUM(ft_grant) AS grant' like this use another name instead.
#     -To filter 'Value Recovery', 'Circular Design and Production', 'Circular Use', 'Clean-up', 'Digital Mapping', 'Incineration', 'Managed Landfilling', 'Operational Platforms', 'Other Services', 'Plastic Waste-to-Energy', 'Plastic Waste-to-Fuel', 'Recovery for controlled disposal', 'Research and Development' and 'Training/ Capacity Building/ Education', always use the 'archetype' column.
#     -In case of asking of program/project description as a general quesiton, you have to use 'sector_name' and 'fund_name' columns and get the answer.
#     -In case of asking of Fund Category as a general quesiton, you have to use 'fund_type' column and get the answer.
#     -To filter 'Africa', 'Asia', 'Europe', 'Latin America And The Caribbean', 'Oceania' and 'North America', always use the 'region' column.
#     -To filter 'Multi Donor National', 'Multilateral' and 'Multi Donor Regional', always use the 'fund_type' column.
#     -To filter 'Adaptation for Smallholder Agriculture Programme (ASAP)', 'Adaptation Fund (AF)', 'Amazon Fund', 'Forest Investment Program (FIP)', 'Global Climate Change Alliance (GCCA)', 'Global Energy Efficiency and Renewable Energy Fund (GEEREF)' and 'Global Environment Facility (GEF4)', always use the 'fund_name' column.
#     -Unique value of 'sids', 'lldc', 'fcs' and 'ida' are '0' and '1'
#     -To check IDA eligible countries need to filter always '1' from 'ida' column.
#     -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', always use the 'commitment' column.
#     -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', In case of asking of ODA as a general quesiton, you have to get all 1 values for all 7 columns always and get the answer.
#     -In case of asking of breakdown of one of sub_category as a general quesiton, you have to use 'sources_intermediaries' column and get the answer.
#     -In case of asking of type of financial flow as a general quesiton, you have to use 'sub_category' column and get the answer.
    
#     For example:
#     Question: How much funds are committed by the Amazon Fund?
#     SQL Query: SELECT SUM(commitment) FROM finances WHERE fund_name = 'Amazon Fund';
    
#     Your turn:
    
#     Question: {question}
#     SQL Query:
#     """


    
#   prompt = ChatPromptTemplate.from_template(template)
  
#   llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4-0125-preview")

  
#   def get_schema(_):
#     return db.get_table_info()
  
#   return (
#     RunnablePassthrough.assign(schema=get_schema)
#     | prompt
#     | llm
#     | StrOutputParser()
#   )



# def get_response(user_query: str, db: SQLDatabase, chat_history: list):
#     sql_chain = get_sql_chain(db)
    
#     template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
#     Based on the table schema below, question, sql query, and sql response, write a natural language response.
#     - You MUST double check your query before executing it.If you get an error while executing a query,rerun the query and try again.
#     - DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE.
#     <SCHEMA>{schema}</SCHEMA>

#     Conversation History: {chat_history}
#     SQL Query: <SQL>{query}</SQL>
#     User question: {question}
#     SQL Response: {response}
    
#     Please decide if the data should be visualized using one of the following graph types: 'line chart', 'stack bar chart', 'bar chart', 'sankey chart'. 
#     If a graph is required, provide the data in the following formats:

#     - **Line Chart**: Use a list of dictionaries with x and y values:
#       ```python
#       [
#           {{x-axis name}}: date, {{y-axis name}}: value,
#           ...
#       ]
#       ```
#     - **Stack Bar Chart**: Use a list of dictionaries with categories and stacked values:
#       ```python
#       [
#           {{category}}: "Category", {{value1}}: value1, {{value2}}: value2,
#           ...
#       ]
#       ```
#     - **Bar Chart**: Use a list of dictionaries with categories and values:
#       ```python
#       [
#           {{category}}: "Category", {{value}}: value,
#           ...
#       ]
#       ```

#     If the answer for the question is a single value or string, provide a direct explained text answer or
#     If the answer needs a graph also, provide both visual and text answer.
    
#     Answer format:
#     - graph_needed: "yes" or "no"
#     - graph_type: one of ['line chart', 'stack bar chart', 'bar chart', 'sankey chart'] (if graph_needed is "yes")
#     - data_array: python data list (if graph_needed is "yes")
#     - text_answer: The direct answer (if graph_needed is "no")
#     """
# # If SQL query is 'None', Say "This is out of database".
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4-0125-preview")
    
#     try:
#         chain = (
#             RunnablePassthrough.assign(query=sql_chain).assign(
#                 schema=lambda _: db.get_table_info(),
#                 response=lambda vars: db.run(vars["query"]),
#             )
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         return chain.invoke({
#             "question": user_query,
#             "chat_history": chat_history,
#         })
    
#     except Exception as e:
#         return f"Error occurred while generating response: {str(e)}"


# # Function to extract fields using regex
# def extract_response_data(result):
#     # Updated regex patterns
#     graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
#     graph_type_pattern = r'graph_type:\s*(\S.*)'
#     data_array_pattern = r'\[\s*(.*?)\s*\]'

#     # Extract fields
#     graph_needed = re.search(graph_needed_pattern, result)
#     graph_type = re.search(graph_type_pattern, result)
#     data_array = re.search(data_array_pattern, result, re.DOTALL)

#     # Extract and clean values
#     graph_needed_value = graph_needed.group(1) if graph_needed else None
#     graph_type_value = graph_type.group(1).strip().strip('"') if graph_type else None
#     data_array_str = data_array.group(1) if data_array else None

#     text_pattern = r'text_answer:\s*(\S.*)'


#     text_output = re.search(text_pattern, result)

#     text_str = text_output.group(1).strip().strip('"') if text_output else None


#     print("=========== data passed to plot the graph =============")
#     print(graph_needed_value)
#     print(graph_type_value)
#     print(data_array_str)
#     print("=======================================================")

#     if data_array_str:
#         # Clean the data array string and convert it to a Python list
#         data_string = f"[{data_array_str}]" # Replace single quotes with double quotes
#         try:
#             # Convert the string to a list of dictionaries
#             data_array_value = json.loads(data_string)
#           # Convert string to Python list
#         except json.JSONDecodeError:
#             print("Error decoding JSON from data_array.")
#             data_array_value = None
#     else:
#         data_array_value = None

#     return graph_needed_value, graph_type_value, data_array_value,text_str
 
  

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#       AIMessage(content="Hello, I am an AI chatbot specialized in global financial flows directed to tackle plastic pollution. You can ask me specifics about these financial flows, including the time period, the name, institution type, and geography of both the flow provider and recipient, the application of the financial flow, and the flow amount based on multiple types of financial flow, such as loan, equity, or grant."),
#     ]

# load_dotenv()

# st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# st.title("Chat with TCI Investment DB")

# st.session_state.db = db
    
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)

# user_query = st.chat_input("Type a message...")
# store_user_query(user_query, engine2)
# if user_query is not None and user_query.strip() != "":
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)

    

#     sql_chain = get_sql_chain(db)
#     response1 = sql_chain.invoke({
#             "chat_history":db,
#             "question":user_query
#         })
#     print("\n=================================")
#     print("\n",response1)
#     print("\n=================================")

        
#     # json_response = get_response_json(user_query, st.session_state.db, st.session_state.chat_history)
#     # print(json_response)
     

#     with st.chat_message("AI"):
#         response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
#         print("\n=================================")
#         graph_needed, graph_type, data_array, text_answer = extract_response_data(response)
#         print("Graph Needed:", graph_needed)
#         print("Graph Type:", graph_type)
#         print("Data Array:", data_array)
#         print("Text Answer:", text_answer)
#         print("\n=================================")
#         # st.markdown(graph_needed)
#         # st.markdown(graph_type)
#         # st.markdown(data_array)
#         # st.markdown(text_answer)
#         # st.markdown(text_answer)
        
#     # st.session_state.chat_history.append(AIMessage(content=response))
#     if text_answer:
#         st.session_state.chat_history.append(AIMessage(content=text_answer))
#         with st.chat_message("AI"):
#             st.markdown(text_answer)
#     else:
#         fallback_message = "I couldn't generate a valid response. Please try again."
#         st.session_state.chat_history.append(AIMessage(content=fallback_message))
#         with st.chat_message("AI"):
#             st.markdown(fallback_message)




###################################################################################################################
######################New App with Function Calling################################################################
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from sqlalchemy import create_engine
import json
import re
from typing import Tuple, List,Dict,Any
import sqlite3
print(sqlite3.sqlite_version)

import chromadb
# from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr
from typing import List


from sqlalchemy import Column, Integer, String, Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker

import os
import pandas as pd
from sqlalchemy import Column, Integer, Text, DateTime
import json
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

##################################################################################################3
#RAG part
chroma_client = chromadb.HttpClient(host='13.235.75.2', port=8000)
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
#End RAG part
############################################################################################################



# # Create a SQLAlchemy engine
# engine = create_engine(os.getenv('DB_CONN_STRING'))
# engine2 = create_engine(os.getenv('DB_CONN_STRING2'))
# # Wrap the engine with SQLDatabase
# db = SQLDatabase(engine)

# Load OpenAI API key
OPENAI_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


# Create a SQLAlchemy engine
#engine = create_engine(os.getenv('DB_CONN_STRING'))
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
database_store = 'investment_tci_prompt'
engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

def _strip(text: str) -> str:
    return text.strip()


from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class UserQuery(Base):
    __tablename__ = 'user_queries'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

# Create the table
Base.metadata.create_all(engine2)

def store_user_query(query: str, engine):
    if not query:  # Check if query is None or an empty string
        print("Error: Query cannot be empty.")
        return
    
    session = sessionmaker(bind=engine)()
    new_query = UserQuery(user_query=query, timestamp=datetime.now())
    session.add(new_query)
    session.commit()
    session.close()

#  Question is not about plastic pollution give response as 'That information is not captured by my database'.

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.This dataset consists of the global financial flows from both public and private sectors directed to tackle plastic pollution. The dataset covers multiple data points for each financial flow, including the time period, the name, institution type, and geography of both the flow provider and recipient, the application of the financial flow, and the flow amount based on multiple types of financial flow, such as loan, equity, or grant.
    Based on the table schema below, write a SQL query that would answer the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)to the database.
  
    Instruction:
    -To filter 'Domestic Finance', 'Development Finance','Private Finance' and 'MEA Finance', always use the 'sub_category' column.
    -To filter 'Circular Pathways', 'Non-circular Pathways' and 'Supporting Pathways', always use the 'pathways' column.
    -There is columns like ft_grant. don't get 'SUM(ft_grant) AS grant' like this use another name instead.
    -To filter 'Value Recovery', 'Circular Design and Production', 'Circular Use', 'Clean-up', 'Digital Mapping', 'Incineration', 'Managed Landfilling', 'Operational Platforms', 'Other Services', 'Plastic Waste-to-Energy', 'Plastic Waste-to-Fuel', 'Recovery for controlled disposal', 'Research and Development' and 'Training/ Capacity Building/ Education', always use the 'archetype' column.
    -In case of asking of program/project description as a general quesiton, you have to use 'sector_name' and 'fund_name' columns and get the answer.
    -In case of asking of Fund Category as a general quesiton, you have to use 'fund_type' column and get the answer.
    -To filter 'Africa', 'Asia', 'Europe', 'Latin America And The Caribbean', 'Oceania' and 'North America', always use the 'region' column.
    -To filter 'Multi Donor National', 'Multilateral' and 'Multi Donor Regional', always use the 'fund_type' column.
    -To filter 'Adaptation for Smallholder Agriculture Programme (ASAP)', 'Adaptation Fund (AF)', 'Amazon Fund', 'Forest Investment Program (FIP)', 'Global Climate Change Alliance (GCCA)', 'Global Energy Efficiency and Renewable Energy Fund (GEEREF)' and 'Global Environment Facility (GEF4)', always use the 'fund_name' column.
    -Unique value of 'sids', 'lldc', 'fcs' and 'ida' are '0' and '1'
    -To check IDA eligible countries need to filter always '1' from 'ida' column.
    -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', always use the 'commitment' column.
    -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', In case of asking of ODA as a general quesiton, you have to get all 1 values for all 7 columns always and get the answer.
    -In case of asking of breakdown of one of sub_category as a general quesiton, you have to use 'sources_intermediaries' column and get the answer.
    -In case of asking of type of financial flow as a general quesiton, you have to use 'sub_category' column and get the answer.


    For example:
    Question: What was the total spend towards tackling plastic pollution in Indonesia from 2018 to 2023?
    SQL Query: SELECT SUM(commitment), FROM investment WHERE country = 'Indonesia' AND AND year >= 2018 AND AND year <= 2023 AND application = 'plastic pollution';
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """


    
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4-0125-preview")

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

    # -In case of asking of Private Equity as a general quesiton, you have to use 'ft_equity' column and get the answer.
#  - If SQL query result is 'None', then give text_answer as 'Not include in the Database'
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.You should execute same SQL suery that provided.
    - You MUST double check your query before executing it.If you get an error while executing a query,rerun the query and try again.
    - DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    
    
    Please decide if the data should be visualized using one of the following graph types: 'line chart', 'stack bar chart', 'bar chart', 'sankey chart'. 
    If a graph is required, provide the data in the following formats:

    - **Line Chart**: Use a list of dictionaries with x and y values:
      ```python
      [
          {{x-axis name}}: date, {{y-axis name}}: value,
          ...
      ]
      ```
    - **Stack Bar Chart**: Use a list of dictionaries with categories and stacked values:
      ```python
      [
          {{category}}: "Category", {{value1}}: value1, {{value2}}: value2,
          ...
      ]
      ```
    - **Bar Chart**: Use a list of dictionaries with categories and values:
      ```python
      [
          {{category}}: "Category", {{value}}: value,
          ...
      ]
      ```

    If the answer for the question is a single value or string, provide a direct explained text answer or
    If the answer needs a graph also, provide both visual and text answer.
    
    Answer format:
    - graph_needed: "yes" or "no"
    - graph_type: one of ['line chart', 'stack bar chart', 'bar chart', 'sankey chart'] (if graph_needed is "yes")
    - data_array: python data list (if graph_needed is "yes")
    - text_answer: The direct answer (if graph_needed is "no")
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4-0125-preview")
    
    try:
        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
    
    except Exception as e:
        return f"Error occurred while generating response: {str(e)}"


# Function to extract fields using regex
def extract_response_data(result):
    # Updated regex patterns
    graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
    graph_type_pattern = r'graph_type:\s*(\S.*)'
    data_array_pattern = r'\[\s*(.*?)\s*\]'

    # Extract fields
    graph_needed = re.search(graph_needed_pattern, result)
    graph_type = re.search(graph_type_pattern, result)
    data_array = re.search(data_array_pattern, result, re.DOTALL)

    # Extract and clean values
    graph_needed_value = graph_needed.group(1) if graph_needed else None
    graph_type_value = graph_type.group(1).strip().strip('"') if graph_type else None
    data_array_str = data_array.group(1) if data_array else None

    text_pattern = r'text_answer:\s*(\S.*)'


    text_output = re.search(text_pattern, result)

    text_str = text_output.group(1).strip().strip('"') if text_output else None


    print("=========== data passed to plot the graph =============")
    print(graph_needed_value)
    print(graph_type_value)
    print(data_array_str)
    print("=======================================================")
    print(text_str)

    if data_array_str:
        # Clean the data array string and convert it to a Python list
        data_string = f"[{data_array_str}]" # Replace single quotes with double quotes
        try:
            # Convert the string to a list of dictionaries
            data_array_value = json.loads(data_string)
          # Convert string to Python list
        except json.JSONDecodeError:
            print("Error decoding JSON from data_array.")
            data_array_value = None
    else:
        data_array_value = None

    return graph_needed_value, graph_type_value, data_array_value,text_str
 
  

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#       AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
#     ]

# load_dotenv()

# st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# st.title("Chat with MySQL")

# st.session_state.db = db
    

# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)

# user_query = st.chat_input("Type a message...")
# store_user_query(user_query, engine2)
# if user_query is not None and user_query.strip() != "":
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)

    

#     sql_chain = get_sql_chain(db)
#     response1 = sql_chain.invoke({
#             "chat_history":db,
#             "question":user_query
#         })
#     print("\n=================================")
#     print("\n",response1)
#     print("\n=================================")

        
#     # json_response = get_response_json(user_query, st.session_state.db, st.session_state.chat_history)
#     # print(json_response)
     

#     with st.chat_message("AI"):
#         response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
#         # print("\n=================================")
#         graph_needed, graph_type, data_array, text_answer = extract_response_data(response)
#         # print("\n=================================")
#         # st.markdown(graph_needed)
#         # st.markdown(graph_type)
#         # st.markdown(data_array)
#         # st.markdown(text_answer)
#         # st.markdown(text_answer)
        
#     # st.session_state.chat_history.append(AIMessage(content=text_answer))
#     # Ensure text_answer is not None before adding to chat history
#     if text_answer:
#         st.session_state.chat_history.append(AIMessage(content=text_answer))
#         with st.chat_message("AI"):
#             st.markdown(text_answer)
#     else:
#         fallback_message = "I couldn't generate a valid response. Please try again."
#         st.session_state.chat_history.append(AIMessage(content=fallback_message))
#         with st.chat_message("AI"):
#             st.markdown(fallback_message)


################################################################################################


###Function Calling Part

# Initialize database connections
OPENAI_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')

# Database setup
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
database_store = 'investment_tci_prompt'
engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")
db = SQLDatabase(engine)


def execute_sql_query(user_query: str, db):
    """Execute database queries for numerical data and statistics."""
    chat_history = [
        AIMessage(content="Hello, I am an AI chatbot specialized in global financial flows directed to tackle plastic pollution. You can ask me specifics about these financial flows."),
        HumanMessage(content=user_query)
    ]

    response = get_response(user_query, db, chat_history)
    graph_needed, graph_type, data_array, text_answer = extract_response_data(response)

    if text_answer:
        return {
            'text_answer': text_answer,
            'graph_needed': graph_needed,
            'graph_type': graph_type,
            'data_array': data_array
        }
    else:
        return {'error': 'No valid response generated'}

def retrieve_from_document(user_query: str):
    """Retrieve definitional, conceptual, and contextual information from documents."""
    response = rag_response(user_query)
    return response

# Function
functions = [
    {
        "name": "execute_sql_query",
        "description": "Use this function for questions about numerical data, statistics, and financial flows. Examples: funding amounts, project counts, temporal trends, geographical distributions,What are the sources of funding? and financial metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The query about numerical data or statistics",
                },
            },
            "required": ["user_query"],
        },
    },
    {
        "name": "retrieve_from_document",
        "description": "Use this function for questions about definitions, concepts, methodologies, and contextual information. Examples: what is a provider, what is public finance, definitions, explain methodologies.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The query about definitions, concepts, or methodology",
                },
            },
            "required": ["user_query"],
        },
    },
]

def get_chatbot_response(user_message: str):
    """Main function to handle user queries and route them to appropriate functions."""
    client = OpenAI(api_key=OPENAI_KEY)
    
    messages = [
        {"role": "system", "content": """You are a specialized financial data assistant. For queries about:
         - Numbers, statistics, funding amounts, trends: use execute_sql_query
         - Definitions, concepts, methodologies: use retrieve_from_document
         Always use one of these functions - don't answer directly."""},
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response = completion.choices[0].message

    if response.function_call:
        function_name = response.function_call.name
        function_args = json.loads(response.function_call.arguments)
        
        if function_name == "execute_sql_query":
            result = execute_sql_query(function_args["user_query"], db)
        elif function_name == "retrieve_from_document":
            result = retrieve_from_document(function_args["user_query"])
            
        return result
    else:
        return {"error": "No function was called"}

def create_plotly_chart(data_array, graph_type):
    """Create a Plotly chart based on the data array and graph type."""
    if not data_array:
        return None
    
    df = pd.DataFrame(data_array)
    
    if graph_type == 'bar':
        fig = px.bar(df, x=df.columns[0], y=df.columns[1])
    elif graph_type == 'line':
        fig = px.line(df, x=df.columns[0], y=df.columns[1])
    elif graph_type == 'pie':
        fig = px.pie(df, values=df.columns[1], names=df.columns[0])
    else:
        fig = px.bar(df, x=df.columns[0], y=df.columns[1]) 
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

# def initialize_session_state():
#     """Initialize session state variables."""
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": {
                    "text_answer": "Hello! I am an AI chatbot specialized in global financial flows directed to tackle plastic pollution. You can ask me specifics about these financial flows, definitions, or methodologies.",
                    "graph_needed": "no",
                    "graph_type": None,
                    "data_array": None
                }
            }
        ]

def main():
      
    st.set_page_config(page_title="The Circuate Initiative Assistant",page_icon=":speech_balloon:")
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title("TCI's Assistant🤖")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write(message["content"]["text_answer"])
                if message["content"].get("graph_needed") == "yes" and message["content"].get("data_array"):
                    fig = create_plotly_chart(
                        message["content"]["data_array"],
                        message["content"]["graph_type"]
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            response = get_chatbot_response(prompt)
            st.write(response["text_answer"])
            
            # Display graph if needed
            if response.get("graph_needed") == "yes" and response.get("data_array"):
                fig = create_plotly_chart(
                    response["data_array"],
                    response["graph_type"]
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
###End of function calling
########################################################################################################
