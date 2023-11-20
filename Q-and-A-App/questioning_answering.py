#!/usr/bin/env python
# coding: utf-8

# # Questioning-and-Answering on Private Documents

# In[1]:


import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# In[2]:


pip install pypdf -q


# In[3]:


pip install docx2txt -q


# In[4]:


pip install wikipedia -q


# ### Loaders

# In[5]:


def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None
    data = loader.load()
    return data

# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


# ### Chunking Data

# In[6]:


def chunk_data(data, chunk_size=256, chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# ### Calculating Cost

# In[7]:


def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


# ### Embedding and Uploading to a Vector Database

# In[8]:


def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Done')
    else:
        print(f'Creating index {index_name} and embeddings ... ', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Done')
    
    return vector_store


# In[9]:


def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Done')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Done')


# ### Asking and Getting Answers

# In[10]:


def ask_and_get_answer(vector_store, query):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(query)
    
    return answer

def ask_with_memory(vector_store, query, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': query, 'chat_history': chat_history})
    chat_history.append((query, result['answer']))
    
    return result, chat_history


# ### Running Code

# In[11]:


data = load_document('./constitution.pdf')
# print(data[1].page_content)
# print(data[10].metadata)
print(f'You have {len(data)} pages in your data.')
print(f'There are {len(data[11].page_content)} characters in the page ')


# In[12]:


data2 = load_document('./World_History.docx')
print(data2[0].page_content)


# In[13]:


data3 = load_from_wikipedia('GPT-4', 'fr')
print(data3[0].page_content)


# In[14]:


chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)


# In[15]:


print_embedding_cost(chunks)


# In[17]:


delete_pinecone_index()


# In[18]:


index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name) 


# In[19]:


query = 'What is the whole document about?'
answer = ask_and_get_answer(vector_store, query)
print(answer)


# In[20]:


import time 
i = 1
print('Write Quit or Exit to quit.')
while True:
    query = input(f'Question #{i}: ')
    i = i + 1
    if query.lower() in ['quit', 'exit']:
        print('Bye Bye ... see you later!')
        time.sleep(2)
        break
    
    answer = ask_and_get_answer(vector_store, query)
    print(f'\nAnswer: {answer}')
    print(f'\n {"-" * 50} \n')


# In[22]:


delete_pinecone_index()


# In[23]:


data = load_from_wikipedia('ChatGPT', 'es')
chunks = chunk_data(data)
index_name = 'chatgpt'
vector_store = insert_or_fetch_embeddings(index_name)


# In[24]:


# query = "Qué es el chat gpt"
# query = "Cuándo se lanzó gpt"
query = "Qué es InstructGPT"
answer = ask_and_get_answer(vector_store, query)
print(answer)


# In[21]:


# asking with memory
chat_history = []
query = "What is the last bill of rights in the US Constitution?"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)


# In[22]:


query = "How many amendments are in the US Constitution"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)


# In[23]:


query = "Multiply that number by 2"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)


# In[ ]:




