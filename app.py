import os
import PyPDF2
import random
import itertools
import streamlit as st
from sqlalchemy import create_engine, text
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings


st.set_page_config(page_title="PDF Analyzer",page_icon=':shark:')


def create_table() -> None:
    conn = st.experimentalconnection("digitalocean", type="sql")
    with conn.session as s:
        # Create the 'knowledge_base' table with specified columns
        s.execute(text("""
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                    id UUID PRIMARY KEY,
                    title VARCHAR() NOT NULL,
                    page_no BIGINT NOT NULL,
                    chunk_no BIGINT NOT NULL,
                    text_chunk TEXT NOT NULL1);"""))


@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0.4))
    eval_set = []
    st.write(sub_sequences)
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


# ...

def main():
    
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 30%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Made by <a href='https://linkedin.com/in/thomasleigh'>Thomas Leigh</a></p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            

        </style>
        """,
        unsafe_allow_html=True,
    )


    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Hettie's Research Assistant</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    
#    embedding_option = "OpenAIEmbeddings"
    create_table()

    embeddings = OpenAIEmbeddings()

    retriever_type = "SIMILARITY SEARCH"

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=200, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings
            # Embed using OpenAI embeddings or HuggingFace embeddings
#        if embedding_option == "OpenAI Embeddings":
        embeddings = OpenAIEmbeddings()
        retriever = create_retriever(embeddings, splits, retriever_type)


        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, model_name="gpt-4", callback_manager=callback_manager, verbose=True, temperature=0.4)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 10  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

       # Display the question-answer pairs in the sidebar with smaller text
#        for i, qa_pair in enumerate(st.session_state.eval_set):
#            st.sidebar.markdown(qa_pair['question'])

        st.divider()
        st.header("Ready to answer questions.")
        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)

        questions_list = [qa_pair['question'] for qa_pair in st.session_state.eval_set]
        answers_list = [qa_pair['answer'] for qa_pair in st.session_state.eval_set]

       # Store the initial value of widgets in session state
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        sample_question = st.selectbox(
            "AI-Generated Questions:",
            questions_list,
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

        if sample_question:
            st.header(sample_question)
            index = questions_list.index(sample_question)
            st.write(answers_list[index])


if __name__ == "__main__":
    main()
