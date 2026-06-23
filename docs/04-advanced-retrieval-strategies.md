# 4. Advanced Retrieval Strategies

These strategies optimize **retrieval quality** -- the most critical factor in RAG performance.

---

## 4.1 Observability and Debugging with LangSmith

### The Problem

In a complex multi-agent RAG system, when an answer is wrong, there is no stack trace. You're debugging by guessing: Was the retrieval bad? Did the LLM ignore the context? Was the prompt template wrong?

### The Solution: LangSmith Traces

LangSmith provides **traces** (every step's input/output), **metrics** (latency, tokens, cost), and **evals** (automated quality testing).

### Setup

```env
# Add to your .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=my-rag-project
```

### Implementation with @traceable

```python
from langsmith.run_helpers import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


@traceable(name="retrieve_documents", tags=["retrieval"])
def retrieve_documents(query: str, k: int = 3):
    """Retrieve relevant documents from the vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs


@traceable(name="format_context", tags=["preprocessing"])
def format_context(docs):
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] ({source}): {doc.page_content}")
    return "\n\n".join(formatted)


@traceable(name="generate_answer", tags=["generation"], metadata={"model": "gpt-4"})
def generate_answer(query: str, context: str):
    """Generate an answer using the LLM with the provided context."""
    prompt = f"""Answer based ONLY on the following context. 
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content


@traceable(
    name="rag_pipeline",
    tags=["production", "rag"],
    metadata={"version": "1.0", "pipeline": "standard"}
)
def rag_pipeline(query: str):
    """Full RAG pipeline with tracing on every step."""
    # Step 1: Retrieve
    docs = retrieve_documents(query)
    
    # Step 2: Format
    context = format_context(docs)
    
    # Step 3: Generate
    answer = generate_answer(query, context)
    
    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc in docs],
        "num_docs_retrieved": len(docs)
    }


# Usage
result = rag_pipeline("What is our refund policy?")
print(result["answer"])
```

### What You See in LangSmith Dashboard

```
rag_pipeline (2.3s, $0.012)
├── retrieve_documents (0.8s)
│   ├── Input: "What is our refund policy?"
│   ├── Output: [3 documents]
│   └── Metadata: k=3
├── format_context (0.001s)
│   ├── Input: [3 documents]
│   └── Output: "[1] (handbook.pdf): Refunds must be..."
└── generate_answer (1.5s, $0.012)
    ├── Input: query + context
    ├── Output: "Our refund policy states that..."
    ├── Tokens: input=850, output=120
    └── Model: gpt-4
```

### Adding User-Level Metadata for Production

```python
@traceable(name="rag_query")
def rag_query(query: str, user_id: str, session_id: str):
    """Production query with user tracking."""
    # LangSmith automatically captures:
    # - Input/output of every step
    # - Token counts and cost
    # - Latency per step
    # - Error traces if something fails
    # - Custom metadata you attach
    
    result = rag_pipeline(query)
    return result

# Track per user
rag_query("refund policy?", user_id="user-123", session_id="sess-abc")
```

---

## 4.2 Parent Document Retriever

### The Problem

There's a fundamental tension in chunking:
- **Small chunks** = precise embeddings = better search accuracy
- **Large chunks** = more context = better LLM answers

You can't have both... unless you decouple search from retrieval.

### The Solution: Search Small, Return Large

```
Index Time:
┌──────────────────────────────────────────────┐
│ Parent Chunk (800 chars)                     │
│ "Authentication uses OAuth 2.0. All requests │
│  must include a Bearer token. Tokens expire  │
│  after 24 hours. Use /auth/refresh to get    │
│  a new token. The refresh token lasts 30     │
│  days. After that, re-authenticate with      │
│  the OAuth2 authorization code flow."        │
│                                              │
│   ┌──────────────┐  ┌──────────────┐         │
│   │ Child (200)  │  │ Child (200)  │ ...     │
│   │ "Auth uses   │  │ "Tokens      │         │
│   │  OAuth 2.0.  │  │  expire      │         │
│   │  Bearer      │  │  after 24h.  │         │
│   │  token..."   │  │  Refresh..." │         │
│   └──────┬───────┘  └──────┬───────┘         │
│          │                 │                  │
│     [embedded]        [embedded]              │
│     [in vector DB]   [in vector DB]           │
└──────────────────────────────────────────────┘

Query Time:
User: "How do I refresh an expired token?"
   │
   ├── Search: Matches CHILD chunk "Tokens expire after 24h. Refresh..."
   │           (small chunk = precise match)
   │
   └── Return: PARENT chunk with full authentication context
               (large chunk = complete context for LLM)
```

### Full Implementation

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ---- Step 1: Define Parent and Child Splitters ----
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # Large chunks for context
    chunk_overlap=100
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,     # Small chunks for precise search
    chunk_overlap=20
)

# ---- Step 2: Create Storage ----
vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)
docstore = InMemoryStore()  # Stores parent-child links

# ---- Step 3: Create the Parent Document Retriever ----
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ---- Step 4: Add Documents ----
loader = PyPDFLoader("api_documentation.pdf")
documents = loader.load()

parent_retriever.add_documents(documents)

# What happens internally:
# 1. Documents are split into parent chunks (800 chars)
# 2. Each parent is split into child chunks (200 chars)
# 3. Child chunks are embedded and stored in the vector store
# 4. A mapping from child_id -> parent_id is stored in the docstore

# ---- Step 5: Query ----
query = "How do I refresh an expired token?"
results = parent_retriever.invoke(query)

# The search happens on CHILD chunks (precise matching)
# But the PARENT chunks are returned (full context)
for doc in results:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content length: {len(doc.page_content)} chars")
    print(f"Content: {doc.page_content[:300]}...")
    print("---")
```

### When to Use Parent Document Retriever

| Scenario | Use It? | Why |
|----------|---------|-----|
| Technical documentation | Yes | Precise search + full procedure context |
| Legal documents | Yes | Precise clause matching + surrounding context |
| Short FAQ pages | No | Chunks are already small enough |
| Code repositories | Maybe | Depends on function/class sizes |

---

## 4.3 Contextual Compression Retriever

### The Problem

Retrieved chunks often contain a lot of **irrelevant noise** mixed with the relevant information. This wastes tokens and can confuse the LLM.

Example:
```
Retrieved Chunk:
"Welcome to Chapter 5 of our API guide. In this chapter, we'll cover 
authentication, rate limiting, and error handling. Let's start with a 
brief history of REST APIs... [200 words of history]... Now, regarding 
authentication: The API key expires after 24 hours. You must refresh 
it using the /auth/refresh endpoint. [100 more words about unrelated topics]"

The ONLY relevant part: "The API key expires after 24 hours. You must 
refresh it using the /auth/refresh endpoint."
```

### The Solution: LLM-Powered Compression

An LLM reads each retrieved chunk and **extracts only the relevant information** before passing it to the generation step.

```
                     Standard Retrieval                    
Query ──> Vector DB ──> [Full Chunk 1] ──> LLM ──> Answer
                        [Full Chunk 2]     ^
                        [Full Chunk 3]     |
                                        lots of noise!

                     Contextual Compression                
Query ──> Vector DB ──> [Full Chunk 1] ──> Compressor ──> [Relevant extract 1] ──> LLM ──> Answer
                        [Full Chunk 2]     (LLM call)     [Relevant extract 2]     ^
                        [Full Chunk 3]                     [Relevant extract 3]     |
                                                                              clean context!
```

### Implementation

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ---- Setup ----
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ---- Step 1: Create the base retriever ----
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---- Step 2: Create the compressor ----
compressor = LLMChainExtractor.from_llm(llm)
# This uses the LLM to extract ONLY the relevant parts of each chunk

# ---- Step 3: Create the compression retriever ----
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# ---- Step 4: Query ----
query = "How do I refresh an expired API token?"

# Without compression
raw_docs = base_retriever.invoke(query)
raw_total_chars = sum(len(doc.page_content) for doc in raw_docs)

# With compression
compressed_docs = compression_retriever.invoke(query)
compressed_total_chars = sum(len(doc.page_content) for doc in compressed_docs)

print(f"Without compression: {raw_total_chars} characters across {len(raw_docs)} docs")
print(f"With compression:    {compressed_total_chars} characters across {len(compressed_docs)} docs")
print(f"Reduction:           {(1 - compressed_total_chars/raw_total_chars)*100:.0f}%")

# Output:
# Without compression: 4,500 characters across 5 docs
# With compression:    675 characters across 5 docs
# Reduction:           85%

# Compare content quality
print("\n--- RAW (first doc) ---")
print(raw_docs[0].page_content[:300])

print("\n--- COMPRESSED (first doc) ---")
print(compressed_docs[0].page_content)
# Only the relevant sentences remain!
```

### Trade-offs

| Aspect | Without Compression | With Compression |
|--------|-------------------|-----------------|
| Token usage | High (lots of noise) | Low (up to 85% reduction) |
| Answer quality | May be confused by noise | Cleaner, more focused |
| Latency | Faster (1 LLM call) | Slower (1 LLM call per chunk + 1 final) |
| Cost per query | Lower | Higher (extra LLM calls for compression) |
| Best for | Simple queries | Complex queries, noisy documents |

### When to Use

- **Use it**: When your documents are verbose and retrieval returns a lot of noise
- **Skip it**: When your chunks are already clean and focused (good chunking solves this better)
- **Consider**: The compression step adds latency (one extra LLM call per retrieved chunk)

---

## Combining Strategies

These strategies can be combined for maximum quality:

```python
# Parent Document Retriever + Contextual Compression
# = Search small, return large, then compress to essentials

from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Step 1: Search small chunks, return parent chunks
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Step 2: Compress the parent chunks to only relevant content
compressor = LLMChainExtractor.from_llm(llm)
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=parent_retriever
)

# Result: Precise search + Full context + Clean extraction
results = final_retriever.invoke("How do I handle token expiration?")
```

### Strategy Selection Guide

```
What's your main problem?
│
├── "Retrieved docs are irrelevant"
│   └── Fix chunking first, then try Parent Document Retriever
│
├── "Retrieved docs are relevant but too noisy"
│   └── Use Contextual Compression Retriever
│
├── "Can't figure out where the pipeline fails"
│   └── Add LangSmith tracing to every step
│
├── "Small chunks miss context, large chunks miss precision"
│   └── Use Parent Document Retriever (search small, return large)
│
└── "All of the above"
    └── Combine: Parent Retriever + Compression + LangSmith tracing
```
