# 3. The 5 Production Failure Modes and Their Fixes

These are the 5 most common ways RAG systems fail in production, and the exact strategies to fix each one.

---

## 3.1 Failure Mode #1: Bad Chunking

### The Problem

Poor chunking creates poor embeddings, which causes poor retrieval. This is the root cause of most RAG failures.

### Symptoms

- Answers are vague or generic
- The LLM says "I don't know" when the answer is clearly in the documents
- Retrieved chunks contain fragments of sentences or mixed topics

### The Fix

Apply all 4 chunking variables correctly (see [02-chunking-strategies.md](02-chunking-strategies.md)):

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# The reliable default for most content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Sweet spot: 200-1000 tokens
    chunk_overlap=50     # "Cheap insurance" against boundary splits
)

# For code
from langchain.text_splitter import Language
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

# For high-quality unstructured text
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)
```

---

## 3.2 Failure Mode #2: Embedding Mismatch

### The Problem

Two common causes:
1. **Different models for indexing and querying**: You embedded documents with `text-embedding-3-small` but query with `text-embedding-ada-002`. The vector spaces are completely different -- search fails silently.
2. **Semantic drift**: The user says "How do I cancel?" but the documents say "Termination policy." The embedding model sees these as different concepts.

### The 3 Golden Rules

| Rule | What It Means |
|------|--------------|
| **Same model, always** | Use the exact same embedding model AND version for indexing and querying |
| **Embedding quality = RAG quality** | 90% of RAG failures are retrieval failures, not LLM failures |
| **Test retrieval separately** | Before blaming the LLM, check what documents are being retrieved |

### Understanding Embeddings

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---- Embedding a single query ----
query_vector = embeddings.embed_query("What is machine learning?")
print(f"Query vector dimensions: {len(query_vector)}")  # 1536

# ---- Embedding multiple documents ----
doc_vectors = embeddings.embed_documents([
    "Machine learning is a subset of AI that learns from data.",
    "The weather in Paris is usually mild in spring.",
    "Deep learning uses neural networks with many layers."
])

# ---- Calculate cosine similarity ----
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print("Similarity scores:")
for i, doc_vec in enumerate(doc_vectors):
    score = cosine_similarity(query_vector, doc_vec)
    print(f"  Doc {i}: {score:.4f}")

# Output:
#   Doc 0: 0.8923  (highly relevant -- about ML)
#   Doc 1: 0.3241  (irrelevant -- about weather)
#   Doc 2: 0.8156  (relevant -- about deep learning, a subset of ML)
```

### Embedding Model Comparison

| Model | Dimensions | Quality | Cost | Best For |
|-------|-----------|---------|------|----------|
| `text-embedding-3-small` | 1536 | Good | $ | Most use cases |
| `text-embedding-3-large` | 3072 | Excellent | $$ | High-accuracy needs |
| Custom dimensions | 512-768 | Configurable | $ | Cost-sensitive apps |

```python
# You can reduce dimensions to save cost and storage
embeddings_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Reduced from 1536 -- saves 66% storage
)
```

### Debugging Retrieval Separately

```python
# ALWAYS test your retrieval before blaming the LLM

# Step 1: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Step 2: Test with a known query
test_query = "What is the refund policy?"
retrieved_docs = retriever.invoke(test_query)

# Step 3: Inspect what was retrieved
print(f"Query: {test_query}")
print(f"Retrieved {len(retrieved_docs)} documents:\n")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print()

# If the retrieved documents are irrelevant, the problem is:
#   - Bad chunking (Fix: adjust chunk_size, overlap, strategy)
#   - Bad embeddings (Fix: try a better embedding model)
#   - Semantic drift (Fix: add hybrid search with BM25)
# If the retrieved documents ARE relevant but the answer is wrong:
#   - The problem is the LLM or prompt (much rarer)
```

---

## 3.3 Failure Mode #3: Retrieval Noise (Hybrid Search)

### The Problem

Vector search (semantic search) fails on **exact-match queries**:
- Product codes: `"SQ7742X"`
- Acronyms: `"WCAG"`, `"GDPR"`
- Error codes: `"ECONREFUSE"`, `"ERR_CONNECTION_TIMEOUT"`
- IDs: `"user-4829-abc"`

The embedding model treats these as meaningless character sequences, not semantic concepts. The vector for `"SQ7742X"` is essentially random.

### The Solution: Hybrid Search

**Hybrid Search = Vector Search (semantic) + BM25 (keyword)**

```
User Query: "What is error SQ7742X?"
      │
      ├──> Vector Search (semantic)          ──> Finds docs about "errors" in general
      │                                           but NOT about SQ7742X specifically
      │
      ├──> BM25 Keyword Search               ──> Finds docs containing "SQ7742X"
      │                                           exact string match
      │
      └──> Reciprocal Rank Fusion (RRF)      ──> Merges both result lists
           Docs that appear in BOTH lists         giving higher scores to docs
           get the highest combined score         that rank well in both searches
```

### Full Implementation

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ---- Step 1: Load and chunk your documents ----
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("technical_manual.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ---- Step 2: Create Vector Retriever (semantic search) ----
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Step 3: Create BM25 Retriever (keyword search) ----
bm25_retriever = BM25Retriever.from_documents(chunks, k=3)

# CRUCIAL PRODUCTION NOTE:
# BM25 does NOT support incremental updates.
# It MUST be rebuilt every time new documents are added.
# If you add new docs to vectorstore, you must also rebuild bm25_retriever.

# ---- Step 4: Combine with Ensemble Retriever ----
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)

# ---- Step 5: Use it ----
results = hybrid_retriever.invoke("What is error SQ7742X?")
for doc in results:
    print(doc.page_content[:150])
    print("---")
```

### Custom Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(result_lists: list[list], k: int = 60):
    """
    Merge multiple ranked result lists using RRF.
    
    RRF score for document d = sum(1 / (k + rank_i)) for each list i
    where rank_i is the rank of d in list i (1-indexed).
    
    Documents appearing in multiple lists get higher combined scores.
    """
    fused_scores = {}
    
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, 1):
            doc_id = doc.page_content  # Use content as unique key
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            # RRF formula: 1 / (k + rank)
            fused_scores[doc_id]["score"] += 1.0 / (k + rank)
    
    # Sort by fused score (highest first)
    sorted_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    return [item["doc"] for item in sorted_results]


# Example usage
vector_results = vector_retriever.invoke("What is error SQ7742X?")
bm25_results = bm25_retriever.invoke("What is error SQ7742X?")

fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
print(f"Top result: {fused_results[0].page_content[:200]}")
```

### Weight Tuning Guide

| Use Case | Vector Weight | BM25 Weight | Reasoning |
|----------|--------------|-------------|-----------|
| General Q&A | 0.7 | 0.3 | Semantics matter more |
| Technical docs with codes | 0.5 | 0.5 | Equal balance |
| Product catalog search | 0.3 | 0.7 | Exact match matters more |
| Legal document search | 0.6 | 0.4 | Semantic + precise terms |

**Performance impact**: Hybrid search adds **20-50ms** of latency but is **low-effort and high-impact** for accuracy.

---

## 3.4 Failure Mode #4: Context Overflow (Token Budgeting)

### The Problem

A single request with a massive document (e.g., a 200-page contract) can blow your daily token budget in one API call.

### The Solution: Token Budget Class

```python
import time
from dataclasses import dataclass, field


@dataclass
class TokenBudget:
    """Track and enforce token usage limits."""
    max_tokens_per_request: int = 4000
    max_tokens_per_day: int = 1_000_000
    max_requests_per_day: int = 10_000
    
    # Tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    daily_reset_time: float = field(default_factory=time.time)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~1.3 tokens per character for English."""
        return int(len(text) * 1.3)
    
    def check_budget(self, estimated_tokens: int) -> dict:
        """Check if a request is within budget BEFORE making the API call."""
        self._maybe_reset_daily()
        
        # Check per-request limit
        if estimated_tokens > self.max_tokens_per_request:
            return {
                "allowed": False,
                "reason": f"Request ({estimated_tokens} tokens) exceeds per-request "
                          f"limit ({self.max_tokens_per_request} tokens)",
                "suggestion": "Reduce input size or increase max_tokens_per_request"
            }
        
        # Check daily limit
        daily_total = self.total_input_tokens + self.total_output_tokens
        if daily_total + estimated_tokens > self.max_tokens_per_day:
            return {
                "allowed": False,
                "reason": f"Would exceed daily limit ({self.max_tokens_per_day} tokens). "
                          f"Used so far: {daily_total}",
                "suggestion": "Wait for daily reset or increase daily limit"
            }
        
        # Check request count
        if self.request_count >= self.max_requests_per_day:
            return {
                "allowed": False,
                "reason": f"Daily request limit reached ({self.max_requests_per_day})"
            }
        
        return {"allowed": True}
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record actual usage after a successful API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
    
    def _maybe_reset_daily(self):
        """Reset counters if 24 hours have passed."""
        if time.time() - self.daily_reset_time > 86400:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.request_count = 0
            self.daily_reset_time = time.time()
    
    def get_usage_stats(self) -> dict:
        """Get current usage statistics for monitoring."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": (
                (self.total_input_tokens * 0.00015 / 1000) +  # GPT-4 input
                (self.total_output_tokens * 0.0006 / 1000)     # GPT-4 output
            )
        }


# ---- Usage in a RAG pipeline ----
budget = TokenBudget(
    max_tokens_per_request=4000,
    max_tokens_per_day=500_000
)

def query_with_budget(query: str, context: str) -> str:
    """Query the LLM with budget enforcement."""
    full_input = f"Context: {context}\n\nQuestion: {query}"
    estimated = budget.estimate_tokens(full_input)
    
    # Check BEFORE making the API call
    check = budget.check_budget(estimated)
    if not check["allowed"]:
        return f"Request blocked: {check['reason']}"
    
    # Make the actual API call
    response = llm.invoke(full_input)
    
    # Record actual usage
    budget.record_usage(
        input_tokens=response.usage_metadata["input_tokens"],
        output_tokens=response.usage_metadata["output_tokens"]
    )
    
    return response.content

# No API call = no cost!
print(budget.get_usage_stats())
```

---

## 3.5 Failure Mode #5: Hallucination

### The Problem

The LLM generates a plausible but incorrect answer, ignoring the provided context.

### The Fix: Robust Grounding Through Prompt Engineering

```python
# THE critical prompt template -- this is your primary defense
GROUNDED_PROMPT = """You are a helpful assistant that answers questions 
based ONLY on the provided context.

RULES:
1. Answer based ONLY on the following context
2. If the context does not contain the answer, say "I don't know based on the available information"
3. Do NOT make up information or use your general knowledge
4. Cite the relevant source for each claim you make
5. If the context is ambiguous, acknowledge the ambiguity

Context:
{context}

Question: {question}

Answer:"""
```

### The Crucial Insight

Fixing the previous 4 failure modes **dramatically reduces hallucination**:

```
Good Chunking     ──> Chunks contain complete, coherent thoughts
     +
Good Embeddings   ──> Similar concepts map to similar vectors
     +
Hybrid Search     ──> Both semantic AND exact matches are found
     +
Token Budgeting   ──> Context fits within limits, nothing truncated
     =
High-quality, relevant, complete context for the LLM
     =
DRAMATICALLY reduced hallucination
```

### Testing for Hallucination

```python
def test_hallucination_resistance():
    """Test that the system says 'I don't know' for out-of-scope questions."""
    
    # Questions that are NOT in the knowledge base
    out_of_scope_questions = [
        "What is the CEO's favorite color?",
        "When will the company IPO?",
        "What is the meaning of life?",
    ]
    
    for question in out_of_scope_questions:
        response = rag_chain.invoke(question)
        assert "don't know" in response.lower() or "not" in response.lower(), \
            f"HALLUCINATION DETECTED for: '{question}'\nResponse: {response}"
        print(f"PASS: '{question}' -> '{response[:80]}...'")

test_hallucination_resistance()
```
