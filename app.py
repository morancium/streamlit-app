# Bring in deps
import os 
import time
import streamlit as st 
import langchain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
import hashlib
from config import OPENAI_API_KEY

from gptcache.adapter.langchain_models import LangChainLLMs
import warnings

persist_directory="full_db"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
    
# ''' GPTcache
# llm = LangChainLLMs(llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))

# def get_hashed_name(name):
#     return hashlib.sha256(name.encode()).hexdigest()

# def init_gptcache(cache_obj: Cache, llm: str):
#     hashed_llm = get_hashed_name(llm)
#     cache_obj.init(
#         pre_embedding_func=get_prompt,
#         data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
#     )

# langchain.llm_cache = GPTCache(init_gptcache)
# '''

from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def final_output(query,llm=llm):
    doc_search_time_start=time.time()
    k=3
    doc=vectordb.similarity_search(query,k=k)
    print("number of similar docs used: ",k)
    doc_search_time_end=time.time()
    print("Time for doc search: ",doc_search_time_end-doc_search_time_start)
    source_url=[]
    info_from_llm_start=time.time()
    info=chain.run(input_documents=doc, question=query)
    info_from_llm_end=time.time()
    print("Time taken to generate response from context from llm: ",info_from_llm_end-info_from_llm_start)
    for source in doc:
        source_url.append(source.metadata["source"])
    return source_url,info

# App framework
# -------- Frontend----------
st.title('AI Assistant')
prompt = st.text_input('Plug in your prompt here') 

# Show stuff to the screen if there's a prompt
if prompt:
    with get_openai_callback() as cb:
        start = time.time()
        source_url,info=final_output(prompt)    
        end = time.time()
        st.write(info)
        print("total time: ",end-start)
        print(cb)
    with st.expander('Source'): 
        source_url_appending_start=time.time()
        for i in range(len(source_url)):
            prefix=source_url[:i]
            name=source_url[i]
            if (prefix.count(name)<=0):
                st.info(name)
        source_url_appending_end=time.time()
        print("Time taken for appending: ",source_url_appending_end-source_url_appending_start)

print("---------------------------------------------------")
# llm = LangChainLLMs(llm=OpenAI(temperature=0))
# '''
# chain = load_qa_chain(llm, chain_type="stuff")
# time1=time.time()
# query = "what is bubble chart"
# docs=vectordb.similarity_search(query,k=2)
# print(chain.run(input_documents=docs, question=query))
# time2=time.time()
# print(time2-time1) 
# '''