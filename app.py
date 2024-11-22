from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import pickle
import torch
import time
import re


# Cache the embeddings model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

# Cache the LLM
@st.cache_resource
def get_llm():
    return Ollama(
        model="llama2",
        temperature=0,
        num_thread=4,  # Increase threads for faster processing
        num_gpu=1 if torch.cuda.is_available() else 0,  # Use GPU if available
        num_ctx=2048,  # Reduced context window for faster processing
    )

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Optimized separators 
    )
    return text_splitter.split_text(text)

def clean_string(str):
    return re.sub(r'')

def create_vectorstore(chunks, embeddings, store_name):
    #Batch process the embeddings
    batch_size = 32
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(
                texts=batch,
                embedding=embeddings,
                metadatas=[{
                    "source": f"chunk_{j}",
                    "pdf_name": store_name,
                    "chunk_index": j
                } for j in range(len(batch))]
            )
        else:
            batch_vectorstore = FAISS.from_texts(
                texts=batch,
                embedding=embeddings,
                metadatas=[{
                    "source": f"chunk_{j+i}",
                    "pdf_name": store_name,
                    "chunk_index": j+i
                } for j in range(len(batch))]
            )
            vectorstore.merge_from(batch_vectorstore)


    return vectorstore

def create_qa_chain(vectorstore, llm):
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "fetch_k": 5
        }
    )

    # Create custom prompt
    prompt_template = """Answer the question based on the context below. Be concise and direct. If you don't know, say so.

    Context: {context}
    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": False
        },
        return_source_documents=True
    )

def main():
    load_dotenv()
    
    # Initialize embeddings at the start
    embeddings = get_embeddings()
    llm = get_llm()

    #initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #initialize session state for qa chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "processed_pdf" not in st.session_state:
        st.session_state.processed_pdf = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = None        

    with st.sidebar:
        st.header('chat with pdf')   
        pdf = st.file_uploader("upload your PDF file", type='pdf')
        st.write("-----")

        if pdf is not None:
            question = st.text_input("ask a question about your pdf", key="question_input")
            st.write("-----")
    
    if pdf is not None and (st.session_state.processed_pdf != pdf.name):
        try:
            with st.spinner("processing PDF"):
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                chunks = process_text(text)
        
            store_name = pdf.name[:-4]
            
            # Try to load existing vectorstore
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vectorstore = pickle.load(f)
                with st.sidebar:
                    st.write('Embeddings loaded from disk')
            else:
                try:
                    # Create new vectorstore
                    vectorstore = create_vectorstore(chunks, embeddings, store_name)
                    # Save vectorstore
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(vectorstore, f)
                    st.write('Embeddings created and saved to disk')
                except Exception as e:
                    st.error(f"Error creating embeddings: {str(e)}")
                    return

            st.session_state.qa_chain = create_qa_chain(vectorstore, llm)
            st.session_state.processed_pdf = pdf.name
            #st.session_state.chat_history = []

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.qa_chain is not None:
        # Display PDF name
        with st.sidebar:
            st.subheader(f"Current PDF: {st.session_state.processed_pdf}")
            st.write("---")
        
        # Display chat history with full context
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                # Question
                st.markdown("**Question:**")
                st.write(q)
                # Answer
                st.markdown("**Answer:**")
                st.write(a)
                st.write("---")

        if 'question' in locals() and question and question != st.session_state.last_question:
            if question.lower() in ['quit', 'exit','bye']:
                st.write("thank you for using the pdf chat. upload a new pdf or ask more questions")
                st.session_state.chat_history=[]
                st.session_state.qa_chain = None
                st.session_state.processed_pdf=None
                st.session_state.last_question = None
            else:
                try:
                    with st.spinner("generating answer"):
                        start_time = time.time()
                        response = st.session_state.qa_chain.invoke({"query": question})
                        end_time = time.time()

                        if "result" in response:
                            answer = response["result"]

                            #add to chat_history
                            st.session_state.chat_history.append((question, answer))
                            st.session_state.last_question = question

                            # Display new answer
                            st.subheader("New Response")
                            st.markdown("**Question:**")
                            st.write(question)
                            st.markdown("**Answer:**")
                            st.write(answer)

                            st.write(f'Response generated in {end_time - start_time:.2f} seconds')

                            # Display sources
                            with st.expander("View Source Documents"):
                                for i, doc in enumerate(response["source_documents"]):
                                    st.write(f"Source {i+1}:")
                                    st.write(doc.page_content)
                                    st.write("---")
                        else:
                            st.write("No answer found")
                  
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}") 

if __name__ == '__main__':
    main()