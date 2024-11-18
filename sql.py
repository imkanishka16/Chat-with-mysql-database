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

# Load OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
database_store = 'investment_tci_prompt'
engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)


import pandas as pd

# Query to select all records from the table
query = "SELECT * FROM finances"

# Execute the query and fetch all records into a DataFrame
with engine2.connect() as connection:
    df = pd.read_sql(query, connection)

# Display the records
print(df)