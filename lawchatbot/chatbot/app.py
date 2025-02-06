import streamlit as st
import requests

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

# API endpoint configuration
API_URL = "http://localhost:8000/query"

def process_input(user_input):
    """
    Send a POST request with the user input and return the result and sources.
    """
    try:
        response = requests.post(API_URL, json={"text": user_input})
        response.raise_for_status()
        data = response.json()
        return data.get("result"), data.get("sources")
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return None, None

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
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "⚖️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Ask your legal question...")
    if user_input:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Researching..."):
            result, sources = process_input(user_input)
            if result:
                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(result)
                    if sources:
                        source_list = ", ".join(source['source'] for source in sources)
                        st.caption(f"Sources: {source_list}")
                st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
