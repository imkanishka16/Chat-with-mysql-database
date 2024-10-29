# from langchain_openai import ChatOpenAI
# from new import get_response, extract_response_data
# from rag import rag_response
# import os
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from langchain_community.utilities import SQLDatabase
# from langchain_core.messages import AIMessage, HumanMessage
# from sqlalchemy import Column, Integer, Text, DateTime
# import json
# from openai import OpenAI

# load_dotenv()

# # Load environment variables
# OPENAI_KEY = os.getenv('OPENAI_KEY')
# user = os.getenv('DB_USER')
# password = os.getenv('DB_PASSWORD') 
# host = os.getenv('DB_HOST')
# database = os.getenv('DB_NAME')

# # Database setup
# engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
# database_store = 'investment_tci_prompt'
# engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")
# db = SQLDatabase(engine)

# def execute_sql_query(user_query: str, db):
#     """
#     Execute database queries for numerical data and statistics.
#     """
#     chat_history = [
#         AIMessage(content="Hello, I am an AI chatbot specialized in global financial flows directed to tackle plastic pollution. You can ask me specifics about these financial flows."),
#         HumanMessage(content=user_query)
#     ]

#     response = get_response(user_query, db, chat_history)
#     graph_needed, graph_type, data_array, text_answer = extract_response_data(response)

#     if text_answer:
#         return {
#             'text_answer': text_answer,
#             'graph_needed': graph_needed,
#             'graph_type': graph_type,
#             'data_array': data_array
#         }
#     else:
#         return {'error': 'No valid response generated'}

# def retrieve_from_document(user_query: str):
#     """
#     Retrieve definitional, conceptual, and contextual information from documents.
#     """
#     response = rag_response(user_query)
#     return {'text_answer': response}

# # Enhanced function definitions with clear distinctions
# functions = [
#     {
#         "name": "execute_sql_query",
#         "description": "Use this function for questions about numerical data, statistics, and financial flows. Examples: funding amounts, project counts, temporal trends, geographical distributions, and financial metrics.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "user_query": {
#                     "type": "string",
#                     "description": "The query about numerical data or statistics",
#                 },
#             },
#             "required": ["user_query"],
#         },
#     },
#     {
#         "name": "retrieve_from_document",
#         "description": "Use this function for questions about definitions, concepts, methodologies, and contextual information. Examples: what is a provider, what is public finance, definitions, explain methodologies.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "user_query": {
#                     "type": "string",
#                     "description": "The query about definitions, concepts, or methodology",
#                 },
#             },
#             "required": ["user_query"],
#         },
#     },
# ]

# def get_chatbot_response(user_message: str):
#     """
#     Main function to handle user queries and route them to appropriate functions.
#     """
#     client = OpenAI(api_key=OPENAI_KEY)
    
#     # Enhanced system message to guide function selection
#     messages = [
#         {"role": "system", "content": """You are a specialized financial data assistant. For queries about:
#          - Numbers, statistics, funding amounts, trends: use execute_sql_query
#          - Definitions, concepts, methodologies: use retrieve_from_document
#          Always use one of these functions - don't answer directly."""},
#         {"role": "user", "content": user_message}
#     ]

#     completion = client.chat.completions.create(
#         model="gpt-4-0613",
#         messages=messages,
#         functions=functions,
#         function_call="auto"
#     )

#     response = completion.choices[0].message

#     # Handle function calls
#     if response.function_call:
#         function_name = response.function_call.name
#         function_args = json.loads(response.function_call.arguments)
        
#         if function_name == "execute_sql_query":
#             result = execute_sql_query(function_args["user_query"], db)
#         elif function_name == "retrieve_from_document":
#             result = retrieve_from_document(function_args["user_query"])
            
#         return result
#     else:
#         return {"error": "No function was called"}

# # Example usage
# if __name__ == "__main__":
#     queries = "What is the total amount of funding received by India from 2021 onwards?"
#     response = get_chatbot_response(queries)
#     print("Response:", response)


import streamlit as st
from langchain_openai import ChatOpenAI
from new import get_response, extract_response_data
from rag import rag_response
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import Column, Integer, Text, DateTime
import json
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

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

# Function definitions (same as your original code)
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

# Function definitions for OpenAI
functions = [
    {
        "name": "execute_sql_query",
        "description": "Use this function for questions about numerical data, statistics, and financial flows. Examples: funding amounts, project counts, temporal trends, geographical distributions, and financial metrics.",
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
        fig = px.bar(df, x=df.columns[0], y=df.columns[1])  # Default to bar chart
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

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
