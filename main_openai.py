
# 1 Load OpenAPI key
import dotenv
dotenv.load_dotenv()

# 2 Build Documents
import requests
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
     f.write(res.text)
from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# 3 Chunkize and Fit into LLM
from langchain.text_splitter import CharacterTextSplitter
chunks = CharacterTextSplitter(
     chunk_size = 500,
     chunk_overlap = 50
).split_documents(
     documents = documents
)

# 4 Build embeddings and store chunk+embedding in Vector db
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
vectorstore = Weaviate.from_documents(
     client = weaviate.Client(
          embedded_options = EmbeddedOptions()
     ),    
     documents = chunks,
     embedding = OpenAIEmbeddings(),
     by_text = False
)

# 5 Retrieve
retriever = vectorstore.as_retriever()

# 6 Augment Prompt with the additional context
from langchain.prompts import ChatPromptTemplate
template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# 7 Generate build a chain for the RAG pipeline, chaining together the retriever, the prompt template and the LLM. 
# Once the RAG chain is defined, you can invoke it.
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
     {"context": retriever,  "question": RunnablePassthrough()} 
     | prompt 
     | llm
     | StrOutputParser() 
)

query = "What did the president say about Justice Breyer"
rag_chain.invoke(query)
