# Bring in deps
import os 
import time
import streamlit as st 
import langchain
from langchain.llms import OpenAI as llm_openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from config import OPENAI_API_KEY

from gptcache import Cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.embedding import OpenAI as cache_openai
from gptcache.adapter.api import init_similar_cache
from langchain.cache import GPTCache
import hashlib
OPENAI_API_KEY=OPENAI_API_KEY
# SQLite
# from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
llm = llm_openai(openai_api_key=OPENAI_API_KEY,temperature=0)
persist_directory="full_db"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)

# GPTCache
openai = cache_openai()
d=openai.dimension
cache_base = CacheBase('sqlite')
vector_base = VectorBase('chromadb', dimension=d)
data_manager = get_data_manager(cache_base, vector_base)

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(embedding=openai ,data_manager=data_manager,cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")

langchain.llm_cache = GPTCache(init_gptcache)
chain = load_qa_chain(llm, chain_type="stuff")

def final_output(query,llm=llm):
    k=3
    doc_search_time_start=time.time()
    doc=vectordb.similarity_search(query,k=k)
    doc_search_time_end=time.time()
    print("number of similar docs used: ",k)
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

# # Show stuff to the screen if there's a prompt
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
# time1=time.time()
# print(llm("who is the ceo of facebook"))
# time2=time.time()
# print("time taken: ", time2-time1)

print("---------------------------------------------------")