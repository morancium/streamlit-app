# Bring in deps
import os 
# from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from config import OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_directory="db"
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
llm = OpenAI(temperature=1, openai_api_key=OPENAI_API_KEY)

chain = load_qa_chain(llm, chain_type="stuff")

def final_output(query,llm=llm):
    doc=vectordb.similarity_search(query)
    # print(doc,"\n")
    source_url=[]
    info=chain.run(input_documents=doc, question=query)
    # print(chain.run(input_documents=doc, question=query))
    # print('\n\nSources:')
    for source in doc:
        source_url.append(source.metadata["source"])
        # print(source.metadata["source"])
    # source_url=list(set(source_url))
    return source_url,info
    # for i in source_url:
    #     print(i)
# query = "what are charts"
# final_output(query)
# App framework


# -------- Frontend----------
st.title('AI Assistant')
prompt = st.text_input('Plug in your prompt here') 

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Show stuff to the screen if there's a prompt
if prompt:
    with get_openai_callback() as cb:
        source_url,info=final_output(prompt)
        st.write(info)
        print(cb)

    with st.expander('Source'): 
        for i in range(len(source_url)):
            prefix=source_url[:i]
            name=source_url[i]
            if (prefix.count(name)<=0):
                st.info(name)