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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings


import os
import pandas as pd

load_dotenv()

# Load OpenAI API key
OPENAI_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


# Create a SQLAlchemy engine
#engine = create_engine(os.getenv('DB_CONN_STRING'))
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

def _strip(text: str) -> str:
    return text.strip()



database_store = 'investment_tci_prompt'
engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")
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
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)to the database.
    If can't generate SQL query for user's question, only 'None'

    Instruction:
    -To filter 'Domestic Finance', 'Development Finance','Private Finance' and 'MEA Finance', always use the 'sub_category' column.
    -To filter 'Circular Pathways', 'Non-circular Pathways' and 'Supporting Pathways', always use the 'pathways' column.
    -To filter 'Value Recovery', 'Circular Design and Production', 'Circular Use', 'Clean-up', 'Digital Mapping', 'Incineration', 'Managed Landfilling', 'Operational Platforms', 'Other Services', 'Plastic Waste-to-Energy', 'Plastic Waste-to-Fuel', 'Recovery for controlled disposal', 'Research and Development' and 'Training/ Capacity Building/ Education', always use the 'archetype' column.
    -In case of asking of program/project description as a general quesiton, you have to use 'sector_name' and 'fund_name' columns and get the answer.
    -In case of asking of Fund Category as a general quesiton, you have to use 'fund_type' column and get the answer.
    -To filter 'Africa', 'Asia', 'Europe', 'Latin America And The Caribbean', 'Oceania' and 'North America', always use the 'region' column.
    -To filter 'Multi Donor National', 'Multilateral' and 'Multi Donor Regional', always use the 'fund_type' column.
    -To filter 'Adaptation for Smallholder Agriculture Programme (ASAP)', 'Adaptation Fund (AF)', 'Amazon Fund', 'Forest Investment Program (FIP)', 'Global Climate Change Alliance (GCCA)', 'Global Energy Efficiency and Renewable Energy Fund (GEEREF)' and 'Global Environment Facility (GEF4)', always use the 'fund_name' column.
    -Unique value of 'sids', 'lldc', 'fcs' and 'ida' are '0' and '1'
    -To check IDA eligible countries need to filter always '1' from 'ida' column.
    -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', always use the 'commitment' column.
    -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', In case of asking of ODA as a general quesiton, you have to get the all 1 values for all 7 columns and get the answer.
    -In case of asking of breakdown of one of sub_category as a general quesiton, you have to use 'sources_intermediaries' column and get the answer. 

    
    For example:
    Question: What was the total spend towards tackling plastic pollution in Indonesia from 2018 to 2023?
    SQL Query: SELECT SUM(Commitment), FROM investment WHERE Country = 'Indonesia' AND AND Year >= 2018 AND AND Year <= 2023 AND Application = 'plastic pollution';
    
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



def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
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
# If SQL query is 'None', Say "This is out of database".
    
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
 
  

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with TCI Investment DB")

st.session_state.db = db
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
store_user_query(user_query, engine2)
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    

    sql_chain = get_sql_chain(db)
    response1 = sql_chain.invoke({
            "chat_history":db,
            "question":user_query
        })
    print("\n=================================")
    print("\n",response1)
    print("\n=================================")

        
    # json_response = get_response_json(user_query, st.session_state.db, st.session_state.chat_history)
    # print(json_response)
     

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        print("\n=================================")
        graph_needed, graph_type, data_array, text_answer = extract_response_data(response)
        print("Graph Needed:", graph_needed)
        print("Graph Type:", graph_type)
        print("Data Array:", data_array)
        print("Text Answer:", text_answer)
        print("\n=================================")
        # st.markdown(graph_needed)
        # st.markdown(graph_type)
        # st.markdown(data_array)
        # st.markdown(text_answer)
        st.markdown(text_answer)
        
    st.session_state.chat_history.append(AIMessage(content=response))
