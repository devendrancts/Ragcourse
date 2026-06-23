# 9. Re-ranking and Cross-Encoders

Re-ranking is one of the highest-leverage improvements you can make to a RAG pipeline. It costs relatively little to implement, adds only modest latency, and consistently delivers 15-25% accuracy improvements. This module explains why re-ranking works, how to implement it, and when to use it.

---

## 9.1 Why Re-ranking Matters

### The Problem with Initial Retrieval

When you retrieve documents using a bi-encoder (standard vector similarity search), you are making a fundamental tradeoff: **speed over precision**. The bi-encoder encodes your query into a single vector, then finds the nearest neighbors in the vector store. This is extremely fast — sub-millisecond for millions of documents — but it is also **approximate**.

The approximation error compounds in several ways:

1. **Semantic compression loss**: Squeezing an entire document chunk into a fixed-size embedding (e.g., 1536 dimensions) inevitably loses nuance.
2. **Query-document interaction is ignored**: The bi-encoder never sees the query and document *together*. It can't reason about how specific tokens in the query relate to specific tokens in the document.
3. **Ranking noise**: The 4th-most-similar vector might actually be more relevant than the 1st — the cosine similarity scores are noisy proxies for true relevance.

The result: your top-3 retrieved chunks are often not your *most relevant* 3 chunks. They are your *most similar-looking* 3 chunks, which is close but not the same thing.

### The Re-ranking Solution

A **cross-encoder re-ranker** takes your top-N retrieved results (e.g., 20) and re-scores each one with a much more powerful relevance model — one that sees the query and the document together. It then returns the true top-K (e.g., 3-5) by relevance.

```
Initial Retrieval (bi-encoder)       Re-ranking (cross-encoder)
─────────────────────────────        ──────────────────────────
Query ──> vector ──> ANN search      Query + Doc1 ──> score: 0.92  ← returned
                     │               Query + Doc2 ──> score: 0.41
                 top-20 docs ──>     Query + Doc3 ──> score: 0.87  ← returned
                                     Query + Doc4 ──> score: 0.78  ← returned
                                     ...
                                     Query + Doc20 ──> score: 0.12
```

### Measured Impact

According to Anthropic's research on contextual retrieval:

| Technique | Retrieval Failure Reduction |
|-----------|----------------------------|
| Baseline (vanilla RAG) | 0% (reference) |
| Contextual embeddings | -35% failures |
| Re-ranking alone | -49% failures |
| Contextual embeddings + re-ranking | -67% failures |

The 67% failure reduction from combining contextual retrieval with re-ranking is one of the most compelling numbers in applied RAG research. Re-ranking alone achieves nearly half of that improvement.

### The Core Tradeoff

| Property | Bi-encoder (Initial Retrieval) | Cross-encoder (Re-ranking) |
|----------|-------------------------------|---------------------------|
| **Speed** | Sub-millisecond per query | 50-200ms for top-20 docs |
| **Scalability** | Millions of documents | Tens of documents only |
| **Accuracy** | Good (approximate) | Excellent (precise) |
| **Use case** | First-pass candidate selection | Final relevance ranking |
| **Token awareness** | None (embeddings only) | Full (token-level interaction) |

---

## 9.2 How Cross-Encoders Work

### Bi-Encoder Architecture (What You're Already Using)

In standard RAG, you use a bi-encoder model like `text-embedding-3-small` or `all-MiniLM-L6-v2`. The architecture is:

```
Query: "What is the refund policy?"
    │
    ▼
[Encoder Model]
    │
    ▼
Query Vector: [0.12, -0.45, 0.78, ...]  (1536 dimensions)


Document: "Customers may return items within 30 days..."
    │
    ▼
[Same Encoder Model]
    │
    ▼
Doc Vector: [0.15, -0.40, 0.81, ...]    (1536 dimensions)


Relevance = cosine_similarity(query_vector, doc_vector) = 0.91
```

The query and document are encoded **independently**. They never interact during encoding. The cosine similarity is computed after the fact. This is what makes it fast — you pre-compute all document vectors and store them.

The limitation: the model compresses "What is the refund policy?" into a single vector *without knowing it will be compared against "Customers may return items within 30 days..."*. Information is lost in that compression.

### Cross-Encoder Architecture (How Re-ranking Works)

A cross-encoder concatenates the query and document into a single input, runs them through a transformer together, and outputs a single scalar relevance score:

```
Input: "[CLS] What is the refund policy? [SEP] Customers may return items within 30 days... [SEP]"
    │
    ▼
[Transformer Model — full attention across BOTH query and document tokens]
    │
    ▼
[CLS] token representation
    │
    ▼
Linear layer
    │
    ▼
Relevance score: 0.94
```

Because the query and document tokens **attend to each other** through the transformer's self-attention mechanism, the model can capture fine-grained interactions:

- The word "refund" in the query attends to "return" in the document (synonyms)
- "policy" attends to "may return items within 30 days" (the policy content)
- "customers" attends to "customers" (exact match reinforcement)

This token-level interaction is what makes cross-encoders so much more accurate. But it also means you **cannot pre-compute anything** — you must run the model at query time for every (query, document) pair. For 20 candidate documents, that's 20 forward passes through the model.

### Why You Can't Use Cross-Encoders for Initial Retrieval

A typical production vector store might contain 500,000 document chunks. Let's compare:

| Approach | Documents | Time per Query |
|----------|-----------|---------------|
| Bi-encoder ANN search | 500,000 | ~1ms |
| Cross-encoder (sequential) | 500,000 | ~500,000 × 10ms = 83 minutes |
| Cross-encoder on top-20 | 20 | ~200ms |

Running a cross-encoder across all 500,000 documents at query time is computationally impossible. The two-stage pipeline is the only practical solution.

### The Two-Stage Pipeline

```
User Query
    │
    ▼
Stage 1: Fast Retrieval (bi-encoder)
    ├── Embed query → vector
    ├── ANN search over vector store
    ├── Retrieve top-20 to top-50 candidates
    └── ~1-5ms total
    │
    ▼
Stage 2: Precise Re-ranking (cross-encoder)
    ├── Score each (query, candidate) pair
    ├── Sort by relevance score descending
    ├── Return top-3 to top-5
    └── ~50-200ms total
    │
    ▼
LLM Generation with top-3 to top-5 highest-quality context chunks
```

The key insight: retrieval fetches **broadly** (top-20 is a wide net), and re-ranking filters **precisely** (top-3 is the cream). You can afford to over-retrieve in stage 1 because stage 2 will clean it up.

---

## 9.3 Implementation with LangChain

### Setup and Dependencies

```python
# Install required packages
# pip install langchain langchain-community sentence-transformers flashrank cohere

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
```

### Option 1: FlashrankRerank (Recommended for Local/Fast Re-ranking)

[Flashrank](https://github.com/PrithivirajDamodaran/FlashRank) is a lightweight re-ranking library that uses ultra-small cross-encoder models. It is very fast (~10-30ms for top-20) and requires no API calls.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Build your base vector store
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# Base retriever — fetches top-20 candidates
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)

# FlashrankRerank compressor — re-ranks to top-5
compressor = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",  # fast and accurate
    top_n=5
)

# Wrap in ContextualCompressionRetriever
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Test it
docs = reranking_retriever.invoke("What is the refund policy?")
for i, doc in enumerate(docs):
    print(f"Rank {i+1}: {doc.page_content[:100]}...")
    print(f"  Score: {doc.metadata.get('relevance_score', 'N/A')}")
```

### Option 2: Sentence-Transformers Cross-Encoder (Most Control)

Using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model directly gives you full control over scoring logic.

```python
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
import numpy as np


class SentenceTransformerReranker:
    """
    Custom re-ranker using sentence-transformers CrossEncoder.
    
    Recommended models (speed vs accuracy tradeoff):
    - cross-encoder/ms-marco-MiniLM-L-6-v2   (fastest, good accuracy)
    - cross-encoder/ms-marco-MiniLM-L-12-v2  (balanced)
    - cross-encoder/ms-marco-electra-base     (slowest, best accuracy)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        score_threshold: float = -10.0  # logit scores; -10 effectively means no threshold
    ):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        self.score_threshold = score_threshold
        print(f"Loaded cross-encoder: {model_name}")

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by relevance to the query.
        Returns list of (document, score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs in a single batched forward pass
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by score descending
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply threshold filter
        filtered = [(doc, score) for doc, score in scored_docs if score > self.score_threshold]

        # Return top-N
        return filtered[:self.top_n]

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank and return only the Document objects (with scores in metadata).
        """
        results = self.rerank(query, documents)
        for doc, score in results:
            doc.metadata["rerank_score"] = float(score)
        return [doc for doc, _ in results]


# Usage
reranker = SentenceTransformerReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5
)

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

query = "How do I cancel my subscription?"

# Stage 1: broad retrieval
candidates = vectorstore.similarity_search(query, k=20)
print(f"Stage 1: Retrieved {len(candidates)} candidates")

# Stage 2: precise re-ranking
reranked = reranker.rerank_documents(query, candidates)
print(f"Stage 2: Re-ranked to top {len(reranked)} documents")

for i, doc in enumerate(reranked):
    print(f"\nRank {i+1} (score: {doc.metadata['rerank_score']:.4f}):")
    print(f"  {doc.page_content[:150]}...")
```

### Option 3: Cohere Rerank API (Best Quality, Cloud-Based)

Cohere's Rerank API is widely regarded as the highest-quality commercial re-ranking option. It requires an API key and charges per request, but delivers state-of-the-art relevance scoring.

```python
import cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Initialize Cohere reranker
cohere_reranker = CohereRerank(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="rerank-english-v3.0",  # or "rerank-multilingual-v3.0" for non-English
    top_n=5
)

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Cohere-powered re-ranking retriever
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=base_retriever
)

# Test
docs = reranking_retriever.invoke("What are the system requirements?")
for i, doc in enumerate(docs):
    print(f"Rank {i+1}: {doc.page_content[:100]}...")
```

**Cohere Rerank Pricing (as of 2025):** ~$1.00 per 1,000 search units (each re-rank of a query + document = 1 search unit). For 20 candidates per query, that's ~$0.02 per query.

### Option 4: Custom Re-ranking Function

For complete control — including custom scoring logic, hybrid scoring, or integration with your own models:

```python
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List, Callable, Optional
import asyncio


class CustomReranker:
    """
    Flexible re-ranker that supports multiple scoring strategies.
    Can be extended with custom business logic (e.g., recency boost,
    source authority weighting, etc.)
    """

    def __init__(
        self,
        primary_scorer: Callable,
        top_n: int = 5,
        recency_boost: bool = False,
        source_weights: Optional[dict] = None
    ):
        self.primary_scorer = primary_scorer
        self.top_n = top_n
        self.recency_boost = recency_boost
        self.source_weights = source_weights or {}

    def _compute_final_score(self, doc: Document, relevance_score: float) -> float:
        """Combine relevance score with optional business logic boosts."""
        score = relevance_score

        # Apply source authority weighting
        source = doc.metadata.get("source", "")
        if source in self.source_weights:
            score *= self.source_weights[source]

        # Apply recency boost (newer docs score higher)
        if self.recency_boost:
            import datetime
            doc_date = doc.metadata.get("date")
            if doc_date:
                days_old = (datetime.date.today() - doc_date).days
                recency_multiplier = max(0.5, 1.0 - (days_old / 365) * 0.3)
                score *= recency_multiplier

        return score

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Score, adjust, sort, and return top-N documents."""
        if not documents:
            return []

        # Get primary relevance scores
        raw_scores = self.primary_scorer(query, documents)

        # Apply business logic adjustments
        final_scores = [
            (doc, self._compute_final_score(doc, score))
            for doc, score in zip(documents, raw_scores)
        ]

        # Sort by final score descending
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Annotate documents with scores and return top-N
        result = []
        for doc, score in final_scores[:self.top_n]:
            doc.metadata["final_rerank_score"] = round(score, 6)
            result.append(doc)

        return result


# Example: use sentence-transformers as the primary scorer
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_scorer(query: str, documents: List[Document]) -> List[float]:
    pairs = [(query, doc.page_content) for doc in documents]
    return cross_encoder.predict(pairs).tolist()


# Custom re-ranker with source authority weighting
reranker = CustomReranker(
    primary_scorer=cross_encoder_scorer,
    top_n=5,
    recency_boost=True,
    source_weights={
        "official_docs": 1.2,   # trust official documentation 20% more
        "community_wiki": 0.9,  # slight penalty for community content
        "forum_post": 0.7       # significant penalty for forum posts
    }
)

# Usage
candidates = vectorstore.similarity_search("How do I reset my password?", k=20)
reranked = reranker.rerank("How do I reset my password?", candidates)
```

### Integration into a Complete RAG Pipeline

This is the end-to-end pattern you should use in production:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List
import time


def format_docs(docs: List) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("relevance_score", "N/A")
        formatted.append(
            f"[Source {i}: {source} | Relevance: {score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def build_reranking_rag_pipeline(
    vectorstore: Chroma,
    llm: ChatOpenAI,
    retrieval_k: int = 20,
    rerank_top_n: int = 5,
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
):
    """
    Build a complete RAG pipeline with re-ranking.
    
    Args:
        vectorstore: Your populated Chroma (or other) vector store
        llm: The language model for generation
        retrieval_k: Number of candidates to retrieve in stage 1
        rerank_top_n: Number of documents to pass to LLM after re-ranking
        rerank_model: Flashrank model name for re-ranking
    
    Returns:
        A LangChain runnable chain
    """

    # Stage 1: broad retrieval
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k}
    )

    # Stage 2: precise re-ranking
    compressor = FlashrankRerank(model=rerank_model, top_n=rerank_top_n)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Answer the question based ONLY on the provided context. If the context does not contain enough information to answer the question, say so explicitly.

Context:
{context}

Question: {question}

Answer:""")

    # Build the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = Chroma(
    collection_name="production_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Build pipeline
chain, retriever = build_reranking_rag_pipeline(
    vectorstore=vectorstore,
    llm=llm,
    retrieval_k=20,
    rerank_top_n=5
)

# Query with timing
query = "What are the steps to configure two-factor authentication?"

start = time.perf_counter()
answer = chain.invoke(query)
elapsed = time.perf_counter() - start

print(f"Answer: {answer}")
print(f"Total latency: {elapsed*1000:.0f}ms")

# Inspect what was retrieved and re-ranked
reranked_docs = retriever.invoke(query)
print(f"\nTop {len(reranked_docs)} documents after re-ranking:")
for i, doc in enumerate(reranked_docs, 1):
    print(f"  {i}. Score: {doc.metadata.get('relevance_score', 'N/A'):.4f} | {doc.page_content[:80]}...")
```

---

## 9.4 Re-ranking Strategies

### Strategy 1: Score-Based Filtering (Threshold Cutoff)

Drop candidates that fall below a minimum relevance score. Prevents injecting irrelevant context into the LLM even when you retrieve K documents.

```python
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import List


class ThresholdReranker:
    """
    Re-ranker that drops documents below a relevance threshold.
    Useful when you'd rather return fewer (or zero) results than
    return irrelevant content to the LLM.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        min_score: float = 0.0,    # cross-encoder logit score; 0.0 is a reasonable floor
        max_results: int = 10      # hard cap regardless of threshold
    ):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        self.min_score = min_score
        self.max_results = max_results

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Score, filter, and sort
        scored = [
            (doc, float(score))
            for doc, score in zip(documents, scores)
            if float(score) >= self.min_score
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Annotate and return
        results = []
        for doc, score in scored[:min(self.top_n, self.max_results)]:
            doc.metadata["rerank_score"] = score
            results.append(doc)

        if not results:
            print(f"Warning: No documents passed the relevance threshold of {self.min_score}")

        return results


# Calibrating your threshold:
# Run your test queries and inspect the score distributions.
# A typical good threshold for ms-marco-MiniLM-L-6-v2:
#   score > 2.0  = very relevant
#   score > 0.0  = probably relevant
#   score < 0.0  = questionable
#   score < -5.0 = likely irrelevant
```

### Strategy 2: Top-K Re-ranking (Standard Pipeline)

The most common pattern: retrieve broadly, re-rank precisely, return a fixed top-K.

```python
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List


def two_stage_retrieve(
    query: str,
    vectorstore: Chroma,
    cross_encoder: CrossEncoder,
    retrieval_k: int = 20,
    rerank_top_n: int = 5
) -> List[Document]:
    """
    Standard two-stage RAG retrieval with top-K re-ranking.
    
    Stage 1: Retrieve retrieval_k candidates via ANN
    Stage 2: Re-rank with cross-encoder, return top rerank_top_n
    """
    import time

    # Stage 1: fast ANN retrieval
    t0 = time.perf_counter()
    candidates = vectorstore.similarity_search(query, k=retrieval_k)
    t1 = time.perf_counter()
    print(f"Stage 1 (retrieval): {len(candidates)} docs in {(t1-t0)*1000:.1f}ms")

    # Stage 2: cross-encoder re-ranking
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    t2 = time.perf_counter()
    print(f"Stage 2 (re-ranking): scored {len(candidates)} docs in {(t2-t1)*1000:.1f}ms")

    # Sort by score and return top-N
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = []
    for doc, score in ranked[:rerank_top_n]:
        doc.metadata["rerank_score"] = float(score)
        results.append(doc)

    print(f"Total retrieval pipeline: {(t2-t0)*1000:.1f}ms")
    return results


# Usage
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

results = two_stage_retrieve(
    query="How do I handle authentication errors?",
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    retrieval_k=20,
    rerank_top_n=5
)
```

### Strategy 3: Diversity-Aware Re-ranking (MMR + Cross-Encoder)

Standard re-ranking can return semantically duplicate results. For example, if you have 10 document chunks all covering the same topic, the cross-encoder might rank them all highly — but they add no new information to the LLM context.

Diversity-aware re-ranking (a variant of Maximal Marginal Relevance) balances relevance with diversity:

```python
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain.schema import Document
from typing import List
import numpy as np


class DiversityAwareReranker:
    """
    Re-ranker that balances relevance (cross-encoder) with diversity
    (bi-encoder similarity between selected documents).
    
    Algorithm: Greedy selection — at each step, pick the document that
    maximizes: lambda * relevance_score - (1 - lambda) * max_similarity_to_selected
    
    lambda = 1.0: pure relevance (standard re-ranking)
    lambda = 0.5: balanced relevance + diversity (recommended)
    lambda = 0.0: pure diversity (not useful in practice)
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bi_encoder_model: str = "all-MiniLM-L6-v2",
        top_n: int = 5,
        diversity_lambda: float = 0.7
    ):
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.top_n = top_n
        self.lambda_ = diversity_lambda

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        if len(documents) <= self.top_n:
            return documents

        # Step 1: Score all documents with cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        relevance_scores = self.cross_encoder.predict(pairs)

        # Normalize relevance scores to [0, 1] for fair comparison
        min_score, max_score = relevance_scores.min(), relevance_scores.max()
        if max_score > min_score:
            norm_scores = (relevance_scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.ones(len(documents))

        # Step 2: Compute bi-encoder embeddings for diversity calculation
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.bi_encoder.encode(doc_texts, normalize_embeddings=True)

        # Step 3: Greedy diverse selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        for _ in range(self.top_n):
            if not remaining_indices:
                break

            best_idx = None
            best_score = float("-inf")

            for idx in remaining_indices:
                relevance = float(norm_scores[idx])

                # Diversity penalty: how similar is this doc to already-selected docs?
                if selected_indices:
                    similarities = [
                        float(np.dot(doc_embeddings[idx], doc_embeddings[sel]))
                        for sel in selected_indices
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0.0

                # MMR score: high relevance, low similarity to selected set
                mmr_score = self.lambda_ * relevance - (1 - self.lambda_) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Annotate and return selected documents
        results = []
        for rank, idx in enumerate(selected_indices):
            doc = documents[idx]
            doc.metadata["rerank_score"] = float(relevance_scores[idx])
            doc.metadata["rerank_rank"] = rank + 1
            results.append(doc)

        return results


# Usage example
reranker = DiversityAwareReranker(
    top_n=5,
    diversity_lambda=0.7  # 70% relevance, 30% diversity
)

candidates = vectorstore.similarity_search("Python async programming", k=20)
diverse_results = reranker.rerank("Python async programming", candidates)

print("Diverse re-ranked results:")
for doc in diverse_results:
    print(f"  Rank {doc.metadata['rerank_rank']}: "
          f"Score={doc.metadata['rerank_score']:.3f} | "
          f"{doc.page_content[:80]}...")
```

### Performance Characteristics

Based on benchmarks using `cross-encoder/ms-marco-MiniLM-L-6-v2` on a standard CPU (Intel i7, no GPU):

| Candidates (K) | Re-ranking Latency | Accuracy vs No Re-ranking |
|---------------|-------------------|--------------------------|
| 10 | ~30ms | +12% |
| 20 | ~60ms | +18% |
| 50 | ~150ms | +22% |
| 100 | ~300ms | +24% |

**Key findings:**
- Diminishing returns beyond top-20 to top-30 candidates
- Re-ranking top-20 is the sweet spot: ~60ms latency, ~18% accuracy gain
- GPU acceleration reduces latency by 5-10x (critical for high-traffic systems)
- Larger cross-encoder models (e.g., `ms-marco-electra-base`) add ~5% more accuracy but 3x more latency

---

## 9.5 When to Use Re-ranking

### Decision Table

| Scenario | Use Re-ranking? | Rationale |
|----------|----------------|-----------|
| Corpus < 1,000 documents | Maybe not | Bi-encoder is already fairly accurate at this scale |
| Corpus > 10,000 documents | Yes | Re-ranking compensates for bi-encoder noise at scale |
| User-facing production system | Yes | 15-25% accuracy gain justifies 50-200ms latency |
| Batch processing / offline pipeline | Yes | Latency doesn't matter; accuracy is everything |
| Real-time autocomplete (<50ms SLA) | No | Cross-encoder latency violates the SLA |
| High query volume (>1,000 QPS) | Consider cloud API | Self-hosted cross-encoder may bottleneck; use Cohere API |
| Multi-lingual content | Yes (use multilingual model) | Bi-encoders struggle more with cross-lingual similarity |
| Domain-specific technical content | Yes | General bi-encoders often misrank technical synonyms |
| Questions with exact keyword answers | Maybe not | BM25 + bi-encoder already strong for keyword queries |
| Complex, nuanced natural language questions | Strongly yes | Cross-encoders shine here |
| Budget-sensitive ($0 LLM/API budget) | Yes, local model | Flashrank + local cross-encoder = free |
| Maximum quality required | Yes + Cohere | Cohere Rerank v3 is currently state-of-the-art |

### Cost vs. Benefit Analysis

#### Self-Hosted Cross-Encoder (Flashrank or sentence-transformers)

| Factor | Value |
|--------|-------|
| Setup cost | 1-2 hours |
| Incremental cost per query | ~$0 (CPU compute) |
| Latency added | 30-200ms depending on model |
| Accuracy improvement | +15-20% |
| GPU required? | No (but helps significantly) |
| Best for | High volume, latency-tolerant, budget-constrained |

#### Cohere Rerank API

| Factor | Value |
|--------|-------|
| Setup cost | 30 minutes |
| Cost per 1,000 queries (top-20 candidates) | ~$20 |
| Latency added | 100-300ms (network + compute) |
| Accuracy improvement | +20-25% (best commercial quality) |
| Best for | Low-to-medium volume, maximum quality required |

#### No Re-ranking (Baseline)

| Factor | Value |
|--------|-------|
| Setup cost | 0 |
| Incremental cost per query | $0 |
| Latency added | 0ms |
| Accuracy vs re-ranking | -15 to -25% |
| Best for | Prototypes, demos, cost-first evaluation |

### Practical Recommendation

For most production RAG systems, the decision tree is:

```
Do you need the best possible answer quality?
│
├── Yes ──> Use re-ranking
│           │
│           ├── Budget available for API?
│           │   ├── Yes ──> Cohere Rerank v3
│           │   └── No  ──> FlashrankRerank (local, free, fast)
│           │
│           └── Query volume?
│               ├── < 100 QPS ──> Single GPU or CPU is sufficient
│               └── > 100 QPS ──> GPU inference server or Cohere API
│
└── No, prototype/demo only ──> Skip re-ranking for now
    (but plan to add it before production launch)
```

### Example: Measuring the Impact in Your Own System

Before committing to re-ranking, measure its actual impact on your specific data and queries:

```python
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import json


def evaluate_reranking_impact(
    test_queries: List[Dict],   # [{"query": str, "relevant_doc_ids": List[str]}]
    vectorstore: Chroma,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    retrieval_k: int = 20,
    rerank_top_n: int = 5
) -> Dict:
    """
    Evaluate the impact of re-ranking on retrieval accuracy.
    
    Computes:
    - Recall@K: fraction of relevant docs found in top-K (before re-ranking)
    - Recall@N: fraction of relevant docs found in top-N (after re-ranking)
    - MRR: Mean Reciprocal Rank (how high is the first relevant doc?)
    
    test_queries format:
        [
            {
                "query": "How do I reset my password?",
                "relevant_doc_ids": ["doc_123", "doc_456"]
            },
            ...
        ]
    """
    cross_encoder = CrossEncoder(cross_encoder_model)

    recall_at_k = []      # before re-ranking
    recall_at_n = []      # after re-ranking
    mrr_before = []       # MRR before re-ranking
    mrr_after = []        # MRR after re-ranking

    for test_case in test_queries:
        query = test_case["query"]
        relevant_ids = set(test_case["relevant_doc_ids"])

        # Stage 1: retrieve top-K
        candidates = vectorstore.similarity_search(query, k=retrieval_k)
        candidate_ids = [doc.metadata.get("doc_id", "") for doc in candidates]

        # Recall@K (before re-ranking)
        found_before = sum(1 for doc_id in candidate_ids if doc_id in relevant_ids)
        recall_at_k.append(found_before / len(relevant_ids) if relevant_ids else 0)

        # MRR before re-ranking
        mrr_b = 0.0
        for rank, doc_id in enumerate(candidate_ids, 1):
            if doc_id in relevant_ids:
                mrr_b = 1.0 / rank
                break
        mrr_before.append(mrr_b)

        # Stage 2: re-rank and get top-N
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_n_ids = [doc.metadata.get("doc_id", "") for doc, _ in ranked[:rerank_top_n]]

        # Recall@N (after re-ranking)
        found_after = sum(1 for doc_id in top_n_ids if doc_id in relevant_ids)
        recall_at_n.append(found_after / len(relevant_ids) if relevant_ids else 0)

        # MRR after re-ranking
        mrr_a = 0.0
        for rank, doc_id in enumerate(top_n_ids, 1):
            if doc_id in relevant_ids:
                mrr_a = 1.0 / rank
                break
        mrr_after.append(mrr_a)

    results = {
        f"recall_at_{retrieval_k}_before_reranking": round(sum(recall_at_k) / len(recall_at_k), 4),
        f"recall_at_{rerank_top_n}_after_reranking": round(sum(recall_at_n) / len(recall_at_n), 4),
        "mrr_before_reranking": round(sum(mrr_before) / len(mrr_before), 4),
        "mrr_after_reranking": round(sum(mrr_after) / len(mrr_after), 4),
        "num_queries_evaluated": len(test_queries)
    }

    recall_improvement = (
        results[f"recall_at_{rerank_top_n}_after_reranking"] -
        results[f"recall_at_{retrieval_k}_before_reranking"]
    ) * 100

    mrr_improvement = (
        results["mrr_after_reranking"] - results["mrr_before_reranking"]
    ) * 100

    print(f"Re-ranking Evaluation Results ({len(test_queries)} queries)")
    print(f"{'─' * 50}")
    print(f"Recall@{retrieval_k} (before): {results[f'recall_at_{retrieval_k}_before_reranking']:.1%}")
    print(f"Recall@{rerank_top_n}  (after):  {results[f'recall_at_{rerank_top_n}_after_reranking']:.1%}")
    print(f"MRR before: {results['mrr_before_reranking']:.4f}")
    print(f"MRR after:  {results['mrr_after_reranking']:.4f}")
    print(f"Recall improvement: {recall_improvement:+.1f}pp")
    print(f"MRR improvement:    {mrr_improvement:+.1f}pp")

    return results


# Example test set
test_queries = [
    {"query": "How do I reset my password?", "relevant_doc_ids": ["doc_001", "doc_002"]},
    {"query": "What payment methods are accepted?", "relevant_doc_ids": ["doc_015"]},
    {"query": "How do I cancel my subscription?", "relevant_doc_ids": ["doc_031", "doc_032", "doc_033"]},
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

results = evaluate_reranking_impact(test_queries, vectorstore)
```

---

## Summary

Re-ranking is a two-stage upgrade to any RAG pipeline that consistently delivers 15-25% accuracy improvements at the cost of 50-200ms additional latency.

| Concept | Key Takeaway |
|---------|-------------|
| **Why re-rank** | Bi-encoder retrieval is approximate; cross-encoders fix the ranking errors |
| **How it works** | Query + document fed together through a transformer for token-level interaction |
| **Can't replace retrieval** | Too slow for millions of docs; only works on top-20 to top-50 candidates |
| **Best local option** | `FlashrankRerank` with `ms-marco-MiniLM-L-12-v2` — fast, free, easy |
| **Best cloud option** | Cohere Rerank v3 — highest accuracy, costs ~$0.02 per query |
| **Combined impact** | Contextual retrieval + re-ranking = 67% reduction in retrieval failures |
| **When to skip** | Prototypes, real-time autocomplete (<50ms SLA), very small corpora |

The next module covers **[10. Evaluation and Metrics for RAG Systems](10-evaluation-and-metrics.md)** — how to measure whether these improvements are actually working.
