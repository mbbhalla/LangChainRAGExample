BEDROCK_FOUNDATION_MODEL_ID_AMAZON_TITAN_V1 = 'amazon.titan-embed-text-v1'
BEDROCK_FOUNDATION_MODEL_ID_ANTHROPIC_CLAUDE_V21 = 'anthropic.claude-v2:1'
REGION = 'us-east-1'
AWS_CREDENTIALS_PROFILE_NAME = 'default'

CONTEXT_URL = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
CONTEXT_FILE_PATH = './CONTEXT_FILE'
PROMPTS = [
     'How much aid is US planning to give to Ukraine?',
     'What countries are part of NATO ?',
     'What is mentioned about infrastructure ?'
]

# Build Documents

import os
if not os.path.exists(CONTEXT_FILE_PATH):
     import requests
     url = CONTEXT_URL
     res = requests.get(url)
     with open(CONTEXT_FILE_PATH, "w") as f:
          f.write(res.text)
from langchain.document_loaders import TextLoader
loader = TextLoader(CONTEXT_FILE_PATH)
documents = loader.load()

# Chunkize and Fit into LLM
from langchain.text_splitter import CharacterTextSplitter
chunks = CharacterTextSplitter(
     chunk_size = 1500,
     chunk_overlap = 250
).split_documents(
     documents = documents
)

# Build embeddings and store chunk+embedding in Vector db
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
vectorstore = Weaviate.from_documents(
     client = weaviate.Client(
          embedded_options = EmbeddedOptions()
     ),    
     documents = chunks,
     embedding = BedrockEmbeddings(
          credentials_profile_name=AWS_CREDENTIALS_PROFILE_NAME, 
          region_name=REGION,
          model_id = BEDROCK_FOUNDATION_MODEL_ID_AMAZON_TITAN_V1
     ),
     by_text = False
)

# Retrieve
retriever = vectorstore.as_retriever()

# Augment Prompt with the additional context
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
prompt_template = ChatPromptTemplate.from_template(template)
print(f"Prompt Template:\n: {prompt_template}")
print("\n")

# Generate build a chain for the RAG pipeline, chaining together the retriever, the prompt template and the LLM. 
#Once the RAG chain is defined, you can invoke it.
from langchain.chat_models import BedrockChat
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = BedrockChat(
     credentials_profile_name=AWS_CREDENTIALS_PROFILE_NAME, 
     region_name=REGION,
     model_id=BEDROCK_FOUNDATION_MODEL_ID_ANTHROPIC_CLAUDE_V21, 
     model_kwargs={"temperature": 1.0}
)
rag_chain = (
     {"context": retriever,  "question": RunnablePassthrough()} 
     | prompt_template
     | llm
     | StrOutputParser() 
)

from termcolor import cprint
for prompt in PROMPTS:
     cprint(f"Prompt: {prompt}\n", "green")
     output = rag_chain.invoke(prompt)
     cprint(f"Output: {output}\n", "light_blue")
     print("\n")

