This repo demonstrates a simple QnA retrieval chain from LangChain. 
The database it uses is collected by doing Web scraping with beautifulsoup library and it is then upserted in the Chromabd vector store.
The openAI LLM uses the chromadb as context to the Questions which has been asked. If it doesn't know the answers, it simply says "I don't know the answer" which is good rather than hallucination.
And a simple frontend is created using StreamLit!
[Website](app.png)