###################################################################################################################
#####################New App with Function Calling################################################################
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# import streamlit as st
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
from datetime import datetime
# import plotly.express as px
# import plotly.graph_objects as go

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')

##################################################################################################
###########################RAG part###############################################################

# chroma_client = chromadb.HttpClient(host='3.110.107.185', port=8000)
chroma_client = chromadb.HttpClient(host='3.110.107.185', port=8000)
chroma_collection = chroma_client.get_collection("tci_glossary")


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


llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")

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
        return answer_text
        # return {"text_answer": answer_text}
    except Exception as e:
        print(f"Error during chain execution: {str(e)}")
        return {"text_answer": "An error occurred while processing your question."}
#End RAG part
############################################################################################################



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
    -When asked of questions by type of funding always refer to 'sub_category' column. 
    -To filter 'Circular Pathways', 'Non-circular Pathways' and 'Supporting Pathways', always use the 'pathways' column.
    -There is columns like ft_grant. don't get 'SUM(ft_grant) AS grant' like this, use another name instead.
    -To filter 'Value Recovery', 'Circular Design and Production', 'Circular Use', 'Clean-up', 'Digital Mapping', 'Incineration', 'Managed Landfilling', 'Operational Platforms', 'Other Services', 'Plastic Waste-to-Energy', 'Plastic Waste-to-Fuel', 'Recovery for controlled disposal', 'Research and Development' and 'Training/ Capacity Building/ Education', always use the 'archetype' column.
    -When asked about program/project description as a general question, you have to use 'sector_name' to get the answer.
    -To filter 'Africa', 'Asia', 'Europe', 'Latin America And The Caribbean', 'Oceania' and 'North America', always use the 'region' column.
    -To filter 'Multi Donor National', 'Multilateral' and 'Multi Donor Regional', always use the 'fund_type' column.
    -To filter by fund name such as  'Adaptation for Smallholder Agriculture Programme (ASAP)', 'Adaptation Fund (AF)', 'Amazon Fund', 'Forest Investment Program (FIP)', 'Global Climate Change Alliance (GCCA)', and 'Global Environment Facility (GEF4)', always use the 'fund_name' column.
    -Unique value of 'sids', 'lldc', 'fcs' and 'ida' are '0' and '1'
    -To check IDA eligible countries need to filter always from 'ida' column. Ids value '1' means eligible and '0' means not eligible. 
    -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', 'investment' or 'commitment' always use the 'financial_flow' column.
    -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', when ask of ODA as a general question, you have to get all 1 values for all 7 columns always and get the answer.

    
    For example:
    Question: Which region received the highest amount of funding in 2022?
    SQL Query: SELECT region, SUM(commitment) AS total_funding FROM finances WHERE year = 2022 GROUP BY region ORDER BY total_funding DESC LIMIT 1;
    Question: What is the split of funding between circular pathways and non-circular pathways?
    SQL Query: SELECT pathways, SUM(commitment) AS total_funding FROM finances WHERE pathways IN ('Circular Pathways', 'Non-circular Pathways') GROUP BY pathways;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-0125-preview")

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )



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
    - graph_type: one of ['line_chart', 'stack_bar_chart', 'bar_chart', 'sankey_chart'] (if graph_needed is "yes", else give 'text')
    - data_array: python data list (if graph_needed is "yes")
    - text_answer: The direct answer (if graph_needed is "no")
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-0125-preview")
    
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


import json
import re

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
        data_string = f"[{data_array_str}]"
        try:
            data_array_value = json.loads(data_string)
        except json.JSONDecodeError:
            print("Error decoding JSON from data_array.")
            data_array_value = None
    else:
        data_array_value = None

    # Process the data to a dynamic format
    if data_array_value and isinstance(data_array_value, list) and len(data_array_value) > 0:
        # Use the first entry to determine label and dataset keys dynamically
        first_entry = data_array_value[0]
        
        # Use any key as a label key if it appears in all entries
        possible_keys = list(first_entry.keys())
        
        # Choose the first available key as label key and use the rest for dataset values
        label_key = possible_keys[0]
        data_keys = possible_keys[1:] if len(possible_keys) > 1 else []
        
        # Extract labels and datasets
        labels = [item.get(label_key, "N/A") for item in data_array_value]
        datasets = [
            tuple(item.get(key, None) for key in data_keys)
            for item in data_array_value
        ]
        
        formatted_data = {
            "labels": labels,
            "datasets": datasets,
            "legend": False
        }
    else:
        formatted_data = {"error": "Data array is empty or not in expected format."}

    return graph_needed_value, graph_type_value, formatted_data, text_str
################################################################################################


###Function Calling Part

# Initialize database connections
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
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
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if text_answer:
        return {
            'provider':'bot',
            'datetime':current_timestamp,
            'type':graph_type,
            'content': text_answer,
            'data': data_array
        }
    else:
        return {
            'provider':'bot',
            'datetime':current_timestamp,
            'type':'error',
            'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
            'data':None
        }


def retrieve_from_document(user_query: str):
    """Retrieve definitional, conceptual, and contextual information from documents."""
    response = rag_response(user_query)
    # Get the current UTC timestamp
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if response:
        return {
            'provider':'bot',
            'datetime':current_timestamp,
            'type':'text',
            'content':response,
            'data': None
        }
    # return response

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

def get_chatbot_response_with_history(user_message: str, chat_history: list):
    """Modified main function to handle user queries with chat history."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Convert chat history to the format OpenAI expects
    messages = [
        {"role": "system", "content": """You are a specialized financial data assistant. For queries about:
         - Numbers, statistics, funding amounts, trends: use execute_sql_query
         - Definitions, concepts, methodologies: use retrieve_from_document
         Always use one of these functions - don't answer directly.
         For follow-up questions, use the context from previous messages to understand what the user is referring to."""}
    ]
    
    # Add chat history
    for msg in chat_history:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"] if msg["role"] == "user" else str(msg["content"])
        })
    
    # Add current message
    messages.append({"role": "user", "content": user_message})

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



# import plotly.express as px
# import streamlit as st
# def create_plotly_chart(data_array, graph_type):
#     """Create a Plotly chart based on the data array and graph type."""
#     if not data_array:
#         return None
    
#     df = pd.DataFrame(data_array)
    
#     if graph_type == 'bar':
#         fig = px.bar(df, x=df.columns[0], y=df.columns[1])
#     elif graph_type == 'line':
#         fig = px.line(df, x=df.columns[0], y=df.columns[1])
#     elif graph_type == 'pie':
#         fig = px.pie(df, values=df.columns[1], names=df.columns[0])
#     else:
#         fig = px.bar(df, x=df.columns[0], y=df.columns[1]) 
        
#     fig.update_layout(
#         margin=dict(l=20, r=20, t=40, b=20),
#         height=400
#     )
#     return fig


# def initialize_session_state():
#     """Initialize session state variables."""
#     if 'messages' not in st.session_state:
#         st.session_state.messages = [
#             {
#                 "role": "assistant", 
#                 "content": {
#                     "text_answer": "Hello! I am an AI chatbot specialized in global financial flows directed to tackle plastic pollution. You can ask me specifics about these financial flows, definitions, or methodologies.",
#                     "graph_needed": "no",
#                     "graph_type": None,
#                     "data_array": None
#                 }
#             }
#         ]


# def main():
      
#     st.set_page_config(page_title="The Circuate Initiative Assistant",page_icon=":speech_balloon:")
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Application header
#     st.title("TCI's Assistant🤖")
    
#     # Chat interface
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             if message["role"] == "assistant":
#                 # st.write(message["content"]["text_answer"])
#                 if message["content"].get("graph_needed") == "yes" and message["content"].get("data_array"):
#                     fig = create_plotly_chart(
#                         message["content"]["data_array"],
#                         message["content"]["graph_type"]
#                     )
#                     if fig:
#                         st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask your question here"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Get bot response
#         with st.chat_message("user"):
#             st.write(prompt)
            
#         with st.chat_message("assistant"):
#             response = get_chatbot_response_with_history(prompt,st.session_state.messages)
#             st.write(response)
            
#             # Display graph if needed
#             if response.get("graph_needed") == "yes" and response.get("data_array"):
#                 fig = create_plotly_chart(
#                     response["data_array"],
#                     response["graph_type"]
#                 )
#                 if fig:
#                     st.plotly_chart(fig, use_container_width=True)

#             sql_chain = get_sql_chain(db)
#             response1 = sql_chain.invoke({
#                     "chat_history":db,
#                     "question":prompt
#                 })
#             print("\n=================================")
#             print("\n",response1)
#             print("\n=================================")
            
#             # Add assistant response to chat history
#             st.session_state.messages.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()
###End of function calling
########################################################################################################


# #############Flask Route###########################
from flask import Flask, request, jsonify, session
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
app.secret_key = 'Abc123@'

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400

        data = request.get_json()
        if 'message' not in data:
            return jsonify({
                'error': 'message field is required'
            }), 400

        # Initialize chat history if it doesn't exist
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Get response with chat history
        response = get_chatbot_response_with_history(data['message'], session['chat_history'])

        # Update chat history
        session['chat_history'].append({
            "role": "user",
            "content": data['message']
        })
        session['chat_history'].append({
            "role": "assistant",
            "content": response
        })

        # Keep only last N messages (e.g., last 10 messages) to prevent session from growing too large
        max_history = 10
        if len(session['chat_history']) > max_history * 2:  # *2 because we store both user and assistant messages
            session['chat_history'] = session['chat_history'][-max_history*2:]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    if 'chat_history' in session:
        session['chat_history'] = []
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


