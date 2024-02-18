import ollama 
import bs4 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 

# loader = WebBaseLoader(
#     web_paths = ('https://mer.vin/'),
#     bs_kwargs=dict(
#         parse_only = bs4.SoupStrainer(
#             class_ = ('post-content', 'post-title', 'post-header')
#         )
#     ),
# )

loader = WebBaseLoader('https://mer.vin/')

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
splits = text_splitter.split_documents(docs)

# create ollama embeddings and vector store
embeddings = OllamaEmbeddings(model='mistral')
vectorstore = Chroma(splits, embeddings)
# vectorstore = Chroma("langchain_store", embeddings)

# create the retriever 
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# define the ollama llm function
# def ollama_llm(question, context):
#     formatted_prompt = f'Question: {question}\n\nContext: {context}'
#     response = ollama.chat(
#         model='mistral',
#         messages = [{'role':'user', 'content':formatted_prompt}]
#     )
#     return response['message','content']

def ollama_llm(question, context):
    formatted_prompt = f'Question: {question}\n\nContext: {context}'
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    return response['messages'][0]['content']

# define the rag chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_content = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

# use the rag chain
result = rag_chain('What is this content is all about?')
print(result)