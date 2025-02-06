from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str
    language: str = "en"

# Initialize resources at startup
@app.on_event("startup")
async def startup_event():
    # Load embeddings
    app.embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Load vector store
    app.vectorstore = FAISS.load_local(
        "vectorstore/db_faiss",
        app.embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM
    app.llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        model_kwargs={"token": os.getenv("HF_TOKEN"), "max_length": 512}
    )
    
    # Configure QA chain
    prompt_template = """[INST] Use context to answer:
    Context: {context}
    Question: {question} 
    Answer directly: [/INST]"""
    
    app.qa_chain = RetrievalQA.from_chain_type(
        llm=app.llm,
        chain_type="stuff",
        retriever=app.vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }
    )

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        response = app.qa_chain.invoke({'query': request.text})
        return {
            "result": response["result"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
