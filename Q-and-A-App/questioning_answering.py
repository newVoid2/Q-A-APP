import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import chroma
import os

# document loader
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

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = chroma.from_documents(chunks, embeddings)
    return vector_store

# Calculating Cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    price = total_tokens / 1000 * 0.0004
    return total_tokens, price

# Asking and Getting Answers
def ask_and_get_answer(vector_store, query, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(query)
    
    return answer


# ### Running Code
data = load_document('./constitution.pdf')
# print(data[1].page_content)
# print(data[10].metadata)
print(f'You have {len(data)} pages in your data.')
print(f'There are {len(data[11].page_content)} characters in the page ')



data2 = load_document('./World_History.docx')
print(data2[0].page_content)



data3 = load_from_wikipedia('GPT-4', 'fr')
print(data3[0].page_content)



chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)


print_embedding_cost(chunks)



delete_pinecone_index()



index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name) 



query = 'What is the whole document about?'
answer = ask_and_get_answer(vector_store, query)
print(answer)


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


delete_pinecone_index()



data = load_from_wikipedia('ChatGPT', 'es')
chunks = chunk_data(data)
index_name = 'chatgpt'
vector_store = insert_or_fetch_embeddings(index_name)



# query = "Qué es el chat gpt"
# query = "Cuándo se lanzó gpt"
query = "Qué es InstructGPT"
answer = ask_and_get_answer(vector_store, query)
print(answer)



# asking with memory
chat_history = []
query = "What is the last bill of rights in the US Constitution?"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)



query = "How many amendments are in the US Constitution"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)



query = "Multiply that number by 2"
result, chat_history = ask_with_memory(vector_store, query, chat_history)
print(result['answer'])
print(chat_history)




