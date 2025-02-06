import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
    <style>
    .stChatMessage {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4b7bff;
    }
    .css-1d391kg {
        background-color: #2d2d2d;
    }
    .stTextInput > div > div > input {
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        border-radius: 8px;
    }
    .stSelectbox > div > div {
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        border-radius: 8px;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ff4b4b, #ff8080);
        height: 8px;
        border-radius: 4px;
    }
    .stSlider > div > div > div > div > div {
        background-color: #ffffff;
        border: 2px solid #ff4b4b;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("⚖️ Legal Assistant")
        st.markdown("Your AI-powered legal research companion")
        st.markdown('</div>', unsafe_allow_html=True)
    
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values generate more precise, factual responses"
        )
        max_length = st.slider(
            "Response Length",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum number of tokens in the response"
        )
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("This AI assistant specializes in legal research and documentation, powered by advanced language models and a comprehensive legal knowledge base.")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "⚖️"):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask me anything...", key="chat_input")

    if prompt:
        # User message
        st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """Answer the question using only the provided context. Be direct and concise.
        Context: {context}
        Question: {question}
        Answer: """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            with st.spinner("Thinking..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
                    return_source_documents=False,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"].strip()

                # Assistant message
                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
