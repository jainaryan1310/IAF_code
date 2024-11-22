# This file contains the REACT Agent

# modules for creating the pandas query engine
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

# modules for setting the llamaindex environment
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as llama_index_ollama

# creating the query engines and the React agent
from llama_index.core import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent

# modules for running the langchain FAISS vector store
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# modules for defining the langchain FAISS RAG CHAIN
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# setting the llamaindex environment
mxbai  = OllamaEmbedding(model_name="mxbai-embed-large")
llama3 = llama_index_ollama(model="llama3")
Settings.embed_model = mxbai
Settings.llm = llama3



# creating the pandas df from the csv
record=pd.read_csv("./data/call_recordings.csv")
ip=pd.read_csv("./data/ip_traffic.csv")
messages=pd.read_csv("./data/messages.csv")


# creating the pandas query engines
instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""


# calls query engine
record_prompt=PromptTemplate.from_template(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
This is the schema of the corresponding dataset
id - Identification Number (int)
date - Date (yyyy-mm-dd)
time - Time (hh:mm:ss)
origin_loc - Starting Location/Origin (str)
dest_loc - Destination (str)
origin_number - Starting Phone number/Origin Phone number (str)
dest_number - Destination Phone Number (str)
call_duration - Call Duration in seconds (int)
link_to_call_recording - These are the links to the voice recordings of the corresponding call.
{instruction_str}
Query: {query_str}

Expression: """
)

call_query_engine=PandasQueryEngine(df=record,instruction_str=instruction_str,pandas_prompt=record_prompt,verbose=True,synthesize_response=True,)


# ip query engine
ip_prompt=PromptTemplate.from_template(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
This is the schema of the corresponding dataset-
id - Identification Number (int)
time - Time taken to send the packets in seconds (float)
source_ip - Source IP address 
destination_ip - Destination IP address 
protocol - Protocol to send the packets (str)
length - Length of the packet (int)
info - Information about the packet (str)
time_difference - Time difference to send the packet (float)
source_frequency - Source Frequency (int)
destination_fequency - Destination Frequency (int)
overlap_frequency - Overlap Frequency (int)
{instruction_str}
Query: {query_str}

Expression: """
)

ip_query_engine=PandasQueryEngine(df=ip,instruction_str=instruction_str,pandas_prompt=ip_prompt,verbose=True,synthesize_response=True)


# message query engine
mess_prompt=PromptTemplate.from_template(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions: 
This is the schema of the corresponding dataset -
id - Identification Number (int)
date - Date (yyyy-mm-dd)
time - Time (hh:mm:ss)
origin_loc - Starting Location/Origin (str)
dest_loc - Destination (str)
origin_number - Starting Phone number / Origin Phone number / Sender Phone number (str)
dest_number - Destination Phone Number / Reciever Phone number (str)
message_text - Message send from the sender to the receiver (str)

If user asks about the messages always give information like time and date when the message was send and also sender phone number and receiver phone number 
{instruction_str}
Query: {query_str}

Expression: """
)

messages_query_engine=PandasQueryEngine(df=messages,instruction_str=instruction_str,pandas_prompt=mess_prompt,verbose=True,synthesize_response=True)


# models to be used by faiss with langchain
ollama_embedding = OllamaEmbeddings(model="mxbai-embed-large")
llm = Ollama(model="llama3")

# loading the FAISS vector store from local storage
db = FAISS.load_local("pdf_vector_store/", ollama_embedding, allow_dangerous_deserialization=True)

# defining the rag chain for the pdf information
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """Answer the below question given the context
question : {question}
context : {context}

If there is no relevant information present in the context, just reply with "I do not know".

response : """

prompt = PromptTemplate.from_template(template)

retriever = db.as_retriever()


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# music system engine
def music_system_engine(query):
    """Answer queries about music systems only and returns the response and sources in a json. Do NOT RESPOND to anythin else"""
    response = rag_chain.invoke(query)
    sources = retriever.invoke(query)
    return {"response": response, "sources": sources}


# defining the function calling functions for the react agent
query_engine_tools=[
    QueryEngineTool(
        query_engine=ip_query_engine,
        metadata=ToolMetadata(
            name="IP_engine",
            description=(
                """
                Provides information about the IP address from where the packets sent and where the received. This also tells about the packets information and the their protocols and also about the frequency.
                """
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=messages_query_engine,
        metadata=ToolMetadata(
            name="Messages_engine",
            description=(
                "Provides information about the sender and reciver of messages"
                "It also consist the messages between sender and reciver"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=call_query_engine,
        metadata=ToolMetadata(
            name="recording_engine",
            description=(
                "Provides links of the recordings of the call between two person. "
                "Provides the location of the callers and the how long they talk"
            ),
        ),
    ),
    FunctionTool.from_defaults(fn=music_system_engine)
]


# system prompt for the react agent 
context="""\
You are the information provider who give information about the call recordings, messages, information Packets and music systems.
1. Use the IP_engine if the question is about ip information or dns
2. For queries about mesasges use the messages_query_engine
3. Use the record_query_engine if the question is about calls
4. Use the music_system_engine if the question is about a music system

If user ask about anything you have to provide all the related information correspondig to that query.
"""


# REACT AGENT
agent = ReActAgent.from_tools(
    query_engine_tools,
    max_iterations=5,
    verbose=False,
    context=context
)






