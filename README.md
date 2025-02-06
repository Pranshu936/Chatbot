# Legal Assistant Chatbot

## Overview

This Legal Assistant Chatbot is an AI-powered application designed to assist with legal research and documentation. It leverages advanced language models and a comprehensive legal knowledge base to provide accurate and relevant information to users' legal queries.

## Features

- **AI-Powered Responses**: Utilizes the Mistral-7B-Instruct-v0.3 model for generating human-like responses to legal questions.
- **Vector Store Integration**: Employs FAISS for efficient retrieval of relevant legal information.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive user experience.
- **Customizable Settings**: Allows users to adjust response creativity and length.
- **Source Attribution**: Provides sources for the information used in responses.
- **PDF and JSONL Document Processing**: Ingests and processes both PDF and JSONL files for comprehensive knowledge base creation.

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Language Model**: HuggingFace's Mistral-7B-Instruct-v0.3
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Language**: Python

## Project Structure

```
pranshu936-chatbot/
├── README.md
└── lawchatbot/
    ├── connect_memory_with_llm.py
    ├── create_memory_for_chatbot.py
    ├── lawchatbot.py
    ├── chatbot/
    │   ├── app.py
    │   ├── api/
    │   │   └── main.py
    │   └── data/
    │       └── data.jsonl
    └── data/
        └── data.jsonl
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pranshu936-chatbot.git
   cd pranshu936-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add:
   ```
   HF_TOKEN=your_huggingface_token
   ```

## Usage

1. Create the vector store:
   ```bash
   python lawchatbot/create_memory_for_chatbot.py
   ```

2. Start the FastAPI backend:
   ```bash
   uvicorn lawchatbot.chatbot.api.main:app --reload
   ```

3. Launch the Streamlit frontend:
   ```bash
   streamlit run lawchatbot/chatbot/app.py
   ```

4. Open your browser and navigate to the provided local URL.

5. Interact with the chatbot by typing your legal questions in the chat input.

## Key Components

### Vector Store Creation (`create_memory_for_chatbot.py`)

- Loads PDF and JSONL files from specified directories.
- Processes and chunks the documents for efficient storage.
- Creates and saves a FAISS vector store with the processed data.

### LLM Integration (`connect_memory_with_llm.py`)

- Sets up the connection between the vector store and the language model.
- Configures the RetrievalQA chain for question answering.

### Main Application (`lawchatbot.py`)

- Implements the core chatbot functionality.
- Manages the user interface and interaction flow.

### API Backend (`chatbot/api/main.py`)

- Provides a FastAPI backend for handling queries.
- Integrates with the LLM and vector store for processing requests.

### Streamlit Frontend (`chatbot/app.py`)

- Offers a user-friendly web interface for interacting with the chatbot.
- Communicates with the backend API to process user queries.

## Customization

The chatbot's behavior can be customized by adjusting parameters such as:

- Temperature (controls response creativity)
- Max length (determines the maximum length of responses)
- Number of retrieved documents (k value in vector store retrieval)

These settings can be modified in the respective Python files or through the Streamlit interface.

## Contributing

Contributions to improve the chatbot are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.
