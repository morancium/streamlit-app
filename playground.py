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
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_directory="db"
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

chain = load_qa_chain(llm, chain_type="stuff")



class Solution:
    def solve(self, nums):
        seen={}
        d={}
        for i in range(len(nums)):
            if not nums[i] in d:
                d[nums[i]]=1
            else:
                d[nums[i]]+=1
        i=0
        while i < len(nums):
            n=d[nums[i]]
            if not nums[i] in seen:
                seen[nums[i]]=1
            else:
                seen[nums[i]]+=1
            if n == seen[nums[i]] and n > 1:
                nums.pop(i)
            i-=1
            i+=1
        return nums
ob = Solution()
# print(ob.solve([10, 30, 40, 10, 30, 50]))
def final_output(query,llm=llm):
    doc=vectordb.similarity_search(query)
    source_url=[]
    info=chain.run(input_documents=doc, question=query)
    for source in doc:
        source_url.append(source.metadata["source"])

    return source_url,info

query = "what is bubble charts"
source_url,info=final_output(query)
print(info)
# source_url=[1,2,1,3,3,4,5,6,1,2,3]
for i in range(len(source_url)):
    prefix=source_url[:i]
    name=source_url[i]
    if (prefix.count(name)<=0):
        print(name)