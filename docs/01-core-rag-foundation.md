# 1. Core RAG Foundation

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge before generating a response. Instead of relying solely on the LLM's training data (which can be outdated or incomplete), RAG retrieves real documents from a knowledge base and feeds them into the prompt.

**Why does this matter?** LLMs like GPT-4 or Claude are powerful but they hallucinate -- they confidently generate plausible-sounding but incorrect answers. RAG *grounds* the LLM's response in actual, retrieved documents, dramatically reducing hallucinations.

---

## 1.1 The Basic RAG Pipeline

The RAG pipeline has 5 core components that work together in sequence:

```
User Query --> Retriever --> Prompt Template --> LLM --> Output Parser --> Final Answer
                  |
            Vector Database
            (your documents)
```

### Component Breakdown

| Component | Role | Example |
|-----------|------|---------|
| **User Query** | Natural language question from the user | "What is our refund policy?" |
| **Retriever** | Searches a vector database for relevant document chunks | Finds 3 chunks about refund policy |
| **Prompt Template** | Combines the query + retrieved context into a structured prompt | "Answer based only on this context: {context}. Question: {query}" |
| **LLM** | Generates the final answer based on the augmented prompt | GPT-4, Claude, etc. |
| **Output Parser** | Parses the LLM's raw output into a structured format | Extracts the string answer |

### Full Implementation Example

```python
# ---- Step 1: Install Dependencies ----
# uv add langchain langchain-openai langchain-chroma chromadb pypdf python-dotenv

# ---- Step 2: Setup ----
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ---- Step 3: Load Documents ----
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ---- Step 4: Split into Chunks ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Each chunk ~500 characters
    chunk_overlap=50      # 50 characters overlap between chunks
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ---- Step 5: Create Embeddings and Store in Vector DB ----
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# ---- Step 6: Create Retriever ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Return top 3 chunks

# ---- Step 7: Define the Prompt Template ----
template = """Answer the question based ONLY on the following context. 
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# ---- Step 8: Create the LLM ----
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ---- Step 9: Build the RAG Chain ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Step 10: Ask a Question ----
response = rag_chain.invoke("What is the refund policy?")
print(response)
```

### How It Works Step by Step

1. **User asks**: "What is the refund policy?"
2. **Retriever**: Converts the question into a vector embedding, searches ChromaDB, and finds the 3 most similar chunks
3. **format_docs**: Joins the 3 chunks into a single string
4. **Prompt Template**: Creates the full prompt:
   ```
   Answer the question based ONLY on the following context.
   If the context does not contain the answer, say "I don't know."
   
   Context:
   Refund requests must be submitted within 30 days of purchase...
   All refunds are processed within 5-7 business days...
   Digital products are non-refundable after download...
   
   Question: What is the refund policy?
   ```
5. **LLM**: Reads the prompt and generates a grounded answer
6. **Output Parser**: Extracts the string response

---

## 1.2 Adding Sources (Citations)

### The Problem

Without citations, users cannot verify the LLM's answer. This reduces trust, especially in enterprise or legal contexts.

### The Solution

Return the **source metadata** (filename, page number) alongside each retrieved chunk, and include it in the prompt so the LLM can cite its sources.

### Implementation

```python
def format_docs_with_sources(docs):
    """Format documents with their source metadata for citation."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# Updated prompt template that asks for citations
template_with_sources = """Answer the question based ONLY on the following context. 
Include citations using [Source N] notation for each fact you reference.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer (with citations):"""

prompt_with_sources = ChatPromptTemplate.from_template(template_with_sources)

# Updated RAG chain
rag_chain_with_sources = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
    | prompt_with_sources
    | llm
    | StrOutputParser()
)

# Example usage
response = rag_chain_with_sources.invoke("What is the refund policy?")
print(response)
# Output: "Refund requests must be submitted within 30 days of purchase [Source 1].
#          All refunds are processed within 5-7 business days [Source 2].
#          Digital products are non-refundable after download [Source 3]."
```

### What the LLM Sees

```
Context:
[Source 1: company_handbook.pdf, Page 12]
Refund requests must be submitted within 30 days of purchase. Customers 
must provide their order number and reason for the refund.

---

[Source 2: company_handbook.pdf, Page 13]
All refunds are processed within 5-7 business days. The refund will be 
credited to the original payment method.

---

[Source 3: company_handbook.pdf, Page 14]
Digital products are non-refundable once the download link has been accessed.
```

---

## 1.3 Environment and Dependency Setup

### Package Manager: `uv`

The course uses `uv` -- a fast Python package manager written in Rust.

```bash
# Initialize a new project
uv init rag-project
cd rag-project

# Create a virtual environment
uv venv

# Activate it
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install all dependencies
uv add langchain langchain-core langgraph
uv add langchain-openai langchain-anthropic
uv add langchain-community langchain-chroma
uv add python-dotenv pypdf rank-bm25
uv add fastapi uvicorn slowapi pydantic-settings
```

### Environment Variables (.env file)

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=my-rag-project
```

### Verification Test

```python
"""verify_setup.py - Run this to confirm everything works."""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Test OpenAI
openai_llm = ChatOpenAI(model="gpt-4")
openai_response = openai_llm.invoke("Say 'setup complete' in one word")
print(f"OpenAI: {openai_response.content}")

# Test Anthropic
anthropic_llm = ChatAnthropic(model="claude-3-sonnet-20240229")
anthropic_response = anthropic_llm.invoke("Say 'setup complete' in one word")
print(f"Anthropic: {anthropic_response.content}")

print("\n--- All systems operational! ---")
```

---

## Key Takeaways

1. **RAG = Retrieval + Generation**: The LLM doesn't guess -- it reads retrieved documents and answers based on them
2. **The prompt template is your primary defense against hallucination**: Always instruct the LLM to say "I don't know" if the context lacks the answer
3. **Citations build trust**: Always return source metadata so users can verify answers
4. **Test retrieval separately**: Before blaming the LLM, check what documents are being retrieved -- 90% of RAG failures are retrieval failures
