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

##################################################################################################3
# #RAG part
# chroma_client = chromadb.HttpClient(host='3.110.107.185', port=8000)
# chroma_collection = chroma_client.get_collection("tci_glossary")

# OPENAI_KEY = os.getenv('OPENAI_KEY')

# class ChromaDBRetriever(BaseRetriever, BaseModel):
#     """Custom retriever for ChromaDB that properly implements Pydantic BaseModel"""
#     _collection: any = PrivateAttr()
#     top_k: int = 3

#     def __init__(self, **data):
#         super().__init__(**data)
#         self._collection = chroma_collection

#     def _get_relevant_documents(self, query: str) -> List[Document]:
#         results = self._collection.query(
#             query_texts=[query],
#             n_results=self.top_k
#         )
#         return [Document(page_content=doc) for doc in results['documents'][0]]

# # Initialize the retriever
# retriever = ChromaDBRetriever()


# llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model="gpt-4o")

# def rag_response(question: str) -> dict:
    
#     template = """You are an assistant for question-answering tasks. 
#     Use the following context to answer the question. If you don't know the answer, just say that you don't know.

#     Context: {context}
#     Question: {question}

#     Provide a clear and direct answer without any JSON formatting or special characters.
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=False,
#         chain_type_kwargs={
#             "prompt": prompt,
#         }
#     )

#     try:
#         raw_result = qa_chain.invoke({"query": question})
#         # Get just the answer text and wrap it in the desired format
#         answer_text = raw_result.get('result', '').strip()
#         return {"text_answer": answer_text}
#     except Exception as e:
#         print(f"Error during chain execution: {str(e)}")
#         return {"text_answer": "An error occurred while processing your question."}
#End RAG part
############################################################################################################

# # Create a SQLAlchemy engine
# engine = create_engine(os.getenv('DB_CONN_STRING'))
# engine2 = create_engine(os.getenv('DB_CONN_STRING2'))
# # Wrap the engine with SQLDatabase
# db = SQLDatabase(engine)

# Load OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
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
    -To filter 'public finance' and 'private finance', always use the 'category' column. You can filter 'private finance' in also 'sub_category'.
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
    -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', 'investment' or 'commitment', always use the 'financial_flow' column.
    -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', when ask of ODA as a general question, you have to get all 1 values for all 7 columns always and get the answer.

    For example:
    1. Question: Based on the last 5 years, trend of funding towards plastic pollution, what do you expect in the next 3 years?
       SQL Query: SELECT pathway, SUM(financial_flow) AS total_funding FROM finances WHERE pathway IN ('Circular Pathways', 'Non-circular Pathways', 'Supporting Pathways') GROUP BY pathway;

    2. Question: What is the split of funding between circular pathways and non-circular pathways?
       SQL Query: SELECT pathways, SUM(financial_flow) AS total_funding FROM finances WHERE pathways IN ('Circular Pathways', 'Non-circular Pathways') GROUP BY pathways;

    3. Question: Which country is the biggest official development assistance provider for tackling plastic pollution in 2021?
       SQL Query: SELECT provider_country, SUM(financial_flow) AS total_funding FROM finances WHERE year = 2021 GROUP BY provider_country ORDER BY total_funding DESC LIMIT 1;

    Your turn:
    
    Question: {question}
    SQL Query:
    """
  
  prompt = ChatPromptTemplate.from_template(template)
  
  model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model
    | StrOutputParser()
  )


# def get_response(user_query: str, db: SQLDatabase, chat_history: list):
#     sql_chain = get_sql_chain(db)
    
#     template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
#     Based on the table schema below, question, sql query, and sql response, write a natural language response.You should execute same SQL suery that provided.
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
#     - graph_type: one of ['line_chart', 'stack_bar_chart', 'bar_chart', 'sankey_chart'] (if graph_needed is "yes")
#     - data_array: python data list (if graph_needed is "yes")
#     - text_answer: The direct answer (if graph_needed is "no")
#     """

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


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company analyzing financial flows for plastic pollution. Based on the provided SQL query and its results, create a clear and accurate response.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    
    Original SQL Query: {query}
    SQL Response: {response}
    User Question: {question}

    Important Instructions:
    1. Use EXACTLY the same SQL query that was generated - DON'T MODIFY IT (warning!!!!)
    2. Base your answer ONLY on the SQL query results - do not use external knowledge
    3. All the values related to finance to be in USD million
    4. For time-based data, describe clear trends
    5. When comparing values, provide relative differences
    
    Visualization Guidelines - ONLY choose one if needed:
    1. Use 'line_chart' for:
       - Time series data
       - Trend analysis over periods
       Example format:
       [
           {{"year": 2020, "funding": 1000000}},
           {{"year": 2021, "funding": 1200000}}
       ]

    2. Use 'stack_bar_chart' for:
       - Comparing parts of a whole
       - Multiple categories over time
       Example format:
       [
           {{"category": "Type A", "value1": 100, "value2": 200}},
           {{"category": "Type B", "value1": 150, "value2": 250}}
       ]

    3. Use 'bar_chart' for:
       - Simple category comparisons
       - Single metric analysis
       Example format:
       [
           {{"region": "Asia", "funding": 500000}},
           {{"region": "Europe", "funding": 700000}}
       ]

    4. Use 'pie_chart' for:
       - Showing proportions of a whole
       - Distribution across categories
       Example format:
       [
           {{"name": "Category A", "value": 1000000}},
           {{"name": "Category B", "value": 2000000}}
       ]

    Your response should follow this format:
    graph_needed: "yes" or "no"
    graph_type: one of ['line_chart', 'stack_bar_chart', 'bar_chart', 'pie_chart', 'text']
    data_array: [your data array if graph is needed]
    text_answer: Your detailed explanation

    Remember: Focus on accuracy and clarity in your response.
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-0125-preview")
    
    try:
        # Get and execute the SQL query
        query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        
        sql_response = db.run(query)
        
        # If no data found
        if not sql_response:
            return """
            graph_needed: no
            graph_type: text
            text_answer: No data found for your query. Please try refining your search criteria.
            """
        
        # Generate the response
        chain = (
            prompt 
            | model 
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "schema": db.get_table_info(),
            "chat_history": chat_history,
            "query": query,
            "response": sql_response,
            "question": user_query
        })
        
        return response
        
    except Exception as e:
        return f"""
        graph_needed: no
        graph_type: text
        text_answer: Error occurred while processing your query: {str(e)}. Please try rephrasing your question.
        """


# Function to extract fields using regex
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
#         print("\n=================================")
#         # st.markdown(graph_needed)
#         # st.markdown(graph_type)
#         # st.markdown(data_array)
#         # st.markdown(text_answer)
#         # st.markdown(response)
        
#     # st.session_state.chat_history.append(AIMessage(content=text_answer))
#     # Ensure text_answer is not None before adding to chat history
#     if text_answer:
#         st.session_state.chat_history.append(AIMessage(content=text_answer))
#         with st.chat_message("AI"):
#             st.markdown(response)
#     else:
#         fallback_message = "I couldn't generate a valid response. Please try again."
#         st.session_state.chat_history.append(AIMessage(content=fallback_message))
#         with st.chat_message("AI"):
#             st.markdown(fallback_message)



from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

app = Flask(__name__)

# In-memory chat history for demo purposes
chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Message field is required'}), 400

        user_query = data['message'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty.'}), 400

        # Prepare chat history for get_response function
        formatted_history = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                return jsonify({"error": "Message content missing in chat history"}), 500
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "AI":
                formatted_history.append(AIMessage(content=content))

        # Generate response with existing function
        response = get_response(user_query, db, formatted_history)
        graph_needed, graph_type, data_array, text_answer = extract_response_data(response)

        # Append messages to chat history
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "AI", "content": text_answer})

        # Keep only last N messages to limit chat history size
        max_history = 10
        if len(chat_history) > max_history * 2:
            del chat_history[:-max_history * 2]

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Return structured JSON response
        return jsonify({
            'provider':'bot',
            'datetime':current_timestamp,
            'type':graph_type,
            'content': text_answer,
            'data': data_array
        }), 200

    except Exception as e:
        return jsonify({
            'provider':'bot',
            'datetime':current_timestamp,
            'type':'error',
            'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
            'data':None
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
