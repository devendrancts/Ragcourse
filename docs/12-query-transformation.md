# 12. Query Transformation Techniques

Query transformation is one of the highest-leverage optimizations in a RAG pipeline. When retrieval fails, the problem is often not the vector database, the chunking strategy, or the embedding model -- it is the query itself.

---

## 12.1 Why Queries Fail

### The Vocabulary Gap

Users describe problems in natural language. Documents describe solutions in technical language. These rarely match.

```
User query:   "how to fix the thing when it won't let me in"
Document:     "Troubleshooting the authentication module: invalid token errors"

Cosine similarity: LOW -- no shared vocabulary
Result: wrong documents retrieved, wrong answer generated
```

This is the **vocabulary gap** -- the mismatch between the words a user chooses and the words used in your knowledge base.

### Common Failure Patterns

| Pattern | Example Query | Why It Fails |
|---------|--------------|-------------|
| Vague pronoun reference | "Why does it crash?" | No subject -- "it" matches everything |
| Domain mismatch | "cancel my thing" vs docs about "subscription termination" | Synonyms not covered |
| Too specific | "Why did the OAuth PKCE flow return 401 on line 47?" | Over-constrained, misses general docs |
| Too broad | "How does the system work?" | Retrieves everything, useful for nothing |
| Misspelling | "autentication errror" | Embeddings are surprisingly tolerant, but edge cases exist |
| Multi-part question | "Compare Plan A and Plan B pricing and features" | Single embedding can't represent two topics |
| Temporal | "What changed last month?" | Vector search has no concept of recency |

### The Fix: Transform Before Retrieval

Rather than hoping the user writes the perfect query, transform the query automatically into a form that retrieves better.

```
Original query ──> Transformation Layer ──> Better query(ies) ──> Retrieval ──> Answer

Techniques:
  - HyDE:               Query → Hypothetical document → Embed that
  - Decomposition:      1 complex query → N simpler queries
  - Step-back:          Specific query → Broader query + original
  - Expansion:          1 query → N synonym variations
  - Multi-query:        LLM generates N diverse phrasings automatically
```

---

## 12.2 HyDE (Hypothetical Document Embeddings)

### The Intuition

When you embed a question like "What is our refund policy?", you get an embedding of a *question*. But your vector store contains embeddings of *answers* -- document passages that describe the refund policy. Questions and answers live in different parts of embedding space.

HyDE (Gao et al., 2022) fixes this with a simple insight: **generate a hypothetical answer, then embed that instead**.

```
Standard retrieval:
"What is our refund policy?" ──[embed]──> query_vector
                                                  │
                              distance to documents (question ≠ answer vocabulary)
                                                  │
                                          [mediocre matches]

HyDE:
"What is our refund policy?" ──[LLM]──> "Customers may request a full refund
                                          within 30 days of purchase. Refunds
                                          are processed within 5-7 business days..."
                                                  │
                                              [embed]
                                                  │
                                          hypothetical_vector
                                                  │
                                distance (answer ≈ answer vocabulary)
                                                  │
                                          [much better matches]
```

The hypothetical answer will be wrong about details -- the LLM is making it up. That does not matter. What matters is that it *sounds like a document*, so it matches document-space embeddings far better than the original question does.

### Basic Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def hyde_retrieval(query: str, k: int = 4) -> list:
    """
    HyDE: embed a hypothetical answer instead of the question.
    """
    # Step 1: Generate a hypothetical answer
    hyde_prompt = f"""Write a short passage (2-4 sentences) that directly answers 
the following question. Write it as if it were an excerpt from a technical document 
or knowledge base article. Do NOT say you don't know -- make a plausible answer.

Question: {query}

Passage:"""

    hypothetical = llm.invoke(hyde_prompt)
    hypothetical_text = hypothetical.content

    # Step 2: Embed the hypothetical answer (not the question)
    docs = vectorstore.similarity_search(hypothetical_text, k=k)

    return docs


# Usage
query = "What is our refund policy?"
docs = hyde_retrieval(query)

for doc in docs:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}")
    print("---")
```

### Full Implementation with Fallback

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time


def hyde_retrieval_with_fallback(
    query: str,
    llm: ChatOpenAI,
    vectorstore: Chroma,
    k: int = 4,
    fallback_to_original: bool = True
) -> dict:
    """
    HyDE retrieval with timing, transparency, and optional fallback.
    
    Returns:
        dict with keys: docs, hypothetical_answer, method, latency_ms
    """
    start = time.time()

    # Step 1: Generate hypothetical answer
    hyde_prompt = f"""Write a passage from a technical knowledge base that answers:

{query}

Write 2-4 sentences as a direct excerpt from documentation. 
Be factual-sounding even if you must approximate.

Passage:"""

    try:
        response = llm.invoke(hyde_prompt)
        hypothetical = response.content.strip()
    except Exception as e:
        if fallback_to_original:
            # Fall back to standard retrieval
            docs = vectorstore.similarity_search(query, k=k)
            return {
                "docs": docs,
                "hypothetical_answer": None,
                "method": "standard_fallback",
                "latency_ms": int((time.time() - start) * 1000),
                "error": str(e)
            }
        raise

    # Step 2: Retrieve using the hypothetical answer
    docs = vectorstore.similarity_search(hypothetical, k=k)

    return {
        "docs": docs,
        "hypothetical_answer": hypothetical,
        "method": "hyde",
        "latency_ms": int((time.time() - start) * 1000)
    }


# Usage
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

result = hyde_retrieval_with_fallback(
    query="How do I reset my password?",
    llm=llm,
    vectorstore=vectorstore,
    k=4
)

print(f"Method: {result['method']}")
print(f"Latency: {result['latency_ms']}ms")
print(f"\nHypothetical answer used for search:")
print(result["hypothetical_answer"])
print(f"\nRetrieved {len(result['docs'])} documents")
```

### When HyDE Helps vs. When It Hurts

| Scenario | HyDE | Why |
|----------|------|-----|
| Question-answering over documentation | Helps significantly | Bridges question-to-document vocabulary gap |
| Short factual queries ("What is X?") | Helps | Generates proper definition-style answer |
| Complex procedural queries | Helps | Generates step-like document content |
| Highly domain-specific (medical, legal) | Use with caution | LLM may hallucinate plausible-sounding but wrong terminology |
| The knowledge base is already Q&A formatted | Minimal gain | Questions already match questions |
| Adversarial/jailbreak-adjacent queries | Hurts | The hypothetical answer may steer into irrelevant space |
| Very short knowledge base (< 50 docs) | Skip | Not worth the extra cost and latency |

### Cost

- HyDE adds **one LLM call per query** before retrieval.
- Using `gpt-4o-mini`: approximately $0.0001-0.0002 per query.
- The embedding call cost is the same -- you are just embedding different text.
- At 10,000 queries/day: ~$1-2/day incremental cost.

---

## 12.3 Query Decomposition

### The Problem

A question like "Compare the pricing and cancellation policy of Plan A versus Plan B" cannot be answered by a single chunk. It requires four distinct pieces of information from potentially four different document sections. A single query embedding will be a blurry compromise that matches none of them well.

### The Solution

Break the complex query into atomic sub-questions. Retrieve and answer each separately. Synthesize a final answer from all sub-answers.

```
Complex query:
"Compare the pricing and features of Plan A vs Plan B"
          │
          ▼
    Decomposition
          │
    ┌─────┴──────┐
    │            │
Sub-question 1: "What is the pricing of Plan A?"
Sub-question 2: "What are the features of Plan A?"
Sub-question 3: "What is the pricing of Plan B?"
Sub-question 4: "What are the features of Plan B?"
    │            │
    └─────┬──────┘
          │
    Retrieve each ──> Answer each ──> Synthesize
```

### Full Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json


llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def decompose_query(query: str, llm: ChatOpenAI) -> list[str]:
    """
    Use an LLM to break a complex question into atomic sub-questions.
    Returns a list of sub-question strings.
    """
    prompt = f"""Break the following question into smaller, self-contained sub-questions.
Each sub-question should be answerable independently from the knowledge base.
Return ONLY a JSON array of strings -- no explanation, no preamble.

Rules:
- Each sub-question must stand alone (no pronouns referring to other sub-questions)
- 2 to 6 sub-questions maximum
- If the question is already simple, return it as a single-element array

Question: {query}

JSON array:"""

    response = llm.invoke(prompt)

    try:
        sub_questions = json.loads(response.content.strip())
        if not isinstance(sub_questions, list):
            return [query]  # Fallback to original
        return sub_questions
    except json.JSONDecodeError:
        return [query]  # Fallback to original


def retrieve_for_subquestion(
    sub_question: str,
    vectorstore: Chroma,
    k: int = 3
) -> list[Document]:
    """Retrieve relevant documents for a single sub-question."""
    return vectorstore.similarity_search(sub_question, k=k)


def answer_subquestion(
    sub_question: str,
    docs: list[Document],
    llm: ChatOpenAI
) -> str:
    """Generate an answer to a sub-question from retrieved docs."""
    context = "\n\n".join(
        f"[Doc {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""Answer the question using ONLY the provided context.
If the context does not contain the answer, say "Not found in knowledge base."
Be concise -- 1-3 sentences.

Context:
{context}

Question: {sub_question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def synthesize_final_answer(
    original_query: str,
    sub_questions: list[str],
    sub_answers: list[str],
    llm: ChatOpenAI
) -> str:
    """Combine all sub-answers into a coherent final answer."""
    qa_pairs = "\n\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in zip(sub_questions, sub_answers)
    )

    prompt = f"""Using the following Q&A pairs, write a comprehensive answer to the 
original question. Synthesize the information -- do not just list the answers.

Original question: {original_query}

Research findings:
{qa_pairs}

Comprehensive answer:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def decomposition_rag(query: str, k_per_subquestion: int = 3) -> dict:
    """
    Full query decomposition RAG pipeline.
    """
    # Step 1: Decompose
    sub_questions = decompose_query(query, llm)
    print(f"Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")

    # Step 2: Retrieve + answer each sub-question
    sub_answers = []
    all_docs = []

    for sub_question in sub_questions:
        docs = retrieve_for_subquestion(sub_question, vectorstore, k=k_per_subquestion)
        all_docs.extend(docs)
        answer = answer_subquestion(sub_question, docs, llm)
        sub_answers.append(answer)

    # Step 3: Synthesize
    final_answer = synthesize_final_answer(query, sub_questions, sub_answers, llm)

    return {
        "answer": final_answer,
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "total_docs_retrieved": len(all_docs),
        "unique_sources": list({
            doc.metadata.get("source", "Unknown") for doc in all_docs
        })
    }


# Usage
result = decomposition_rag(
    "Compare the pricing and cancellation policy of Plan A versus Plan B, "
    "and explain which plan is better for a startup of 10 people."
)

print("\n=== FINAL ANSWER ===")
print(result["answer"])
print(f"\nSources: {result['unique_sources']}")
```

### Deduplicating Retrieved Documents

When multiple sub-questions retrieve overlapping documents, deduplication avoids sending redundant context to the LLM:

```python
def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents by content hash."""
    seen = set()
    unique = []
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    return unique
```

### Cost Considerations

- Decomposition adds one LLM call upfront.
- Each sub-question adds one retrieval + one LLM call.
- For a query decomposed into 4 sub-questions: 6 total LLM calls (1 decompose + 4 answer + 1 synthesize).
- Use `gpt-4o-mini` for decompose/answer steps and `gpt-4o` only for synthesis to control costs.

---

## 12.4 Step-Back Prompting

### The Intuition

Specific questions often fail retrieval because the exact answer lives in a general-purpose section, not a specialized one. A question about "Q3 2024 revenue drop" may be best answered by a document about "factors affecting quarterly revenue" -- a more general treatment.

Step-back prompting (Zheng et al., 2023) retrieves context at two levels: the original specific query and a broader "step-back" version. Combining both gives the LLM specific details *plus* general principles to reason from.

```
Original query:  "Why did revenue drop in Q3 2024?"
Step-back query: "What factors commonly affect quarterly revenue performance?"

Retrieval 1 (specific):  Q3 2024 earnings release, CFO commentary
Retrieval 2 (general):   Revenue modeling guide, financial best practices

Combined context = specific event + general framework = better answer
```

### Full Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def generate_step_back_query(query: str, llm: ChatOpenAI) -> str:
    """
    Generate a broader, more general version of the query.
    """
    prompt = f"""Your task is to generate a broader, more general version of 
a specific question. This step-back question should retrieve background 
concepts, principles, or overviews that help answer the original question.

Examples:
- Specific: "Why did revenue drop in Q3 2024?"
  Step-back: "What factors commonly affect quarterly revenue performance?"

- Specific: "How do I fix error code 403 in the payment API?"
  Step-back: "How does API authorization and permission handling work?"

- Specific: "What is the refund timeline for international orders?"
  Step-back: "How does the refund process work?"

Now generate the step-back question for:
Specific: "{query}"
Step-back:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def step_back_retrieval(
    query: str,
    k_specific: int = 3,
    k_stepback: int = 3
) -> dict:
    """
    Retrieve documents for both the specific query and the step-back query.
    Returns combined, deduplicated context.
    """
    # Generate step-back query
    step_back_query = generate_step_back_query(query, llm)
    print(f"Original:  {query}")
    print(f"Step-back: {step_back_query}")

    # Retrieve for both
    specific_docs = vectorstore.similarity_search(query, k=k_specific)
    stepback_docs = vectorstore.similarity_search(step_back_query, k=k_stepback)

    # Deduplicate (prefer specific docs in ordering)
    seen = set()
    combined_docs = []
    for doc in specific_docs + stepback_docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            combined_docs.append(doc)

    return {
        "combined_docs": combined_docs,
        "specific_docs": specific_docs,
        "stepback_docs": stepback_docs,
        "step_back_query": step_back_query
    }


def step_back_rag(query: str) -> str:
    """Full step-back RAG pipeline."""
    retrieval = step_back_retrieval(query)

    # Build context, labeling which source each doc came from
    specific_context = "\n\n".join(
        f"[Specific context {i+1}]: {doc.page_content}"
        for i, doc in enumerate(retrieval["specific_docs"])
    )
    general_context = "\n\n".join(
        f"[General context {i+1}]: {doc.page_content}"
        for i, doc in enumerate(retrieval["stepback_docs"])
    )

    prompt = f"""You have two types of context to answer the question:
1. Specific context: directly relevant to the question
2. General context: broader background principles

Use both to give a thorough, well-reasoned answer.

--- SPECIFIC CONTEXT ---
{specific_context}

--- GENERAL CONTEXT ---
{general_context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content.strip()


# Usage
answer = step_back_rag("Why did revenue drop in Q3 2024?")
print("\n=== ANSWER ===")
print(answer)
```

### When Step-Back Prompting Shines

- "Why did X fail?" -- the step-back retrieves general failure mode documentation
- "What does error code 403 mean in context Y?" -- step-back retrieves HTTP authorization concepts
- Domain-specific questions where the user knows an instance but the docs explain the category

### Cost

- One extra LLM call to generate the step-back query (cheap with `gpt-4o-mini`).
- One extra retrieval call (negligible cost).
- Total: approximately 1.5-2x the cost of a standard RAG query.

---

## 12.5 Query Expansion

### The Problem

Your users say "cancel", your docs say "terminate". Your users say "end subscription", your docs say "account deactivation". Synonyms cause silent retrieval failures -- the vector store finds nothing wrong, but returns the wrong documents.

### The Solution

Generate multiple variations of the query and retrieve documents for each. Then merge results using **Reciprocal Rank Fusion (RRF)** -- a ranking algorithm that rewards documents that appear in the top results of multiple queries.

```
"how do I cancel my subscription"
          │
    Query Expansion
          │
    ┌─────┼─────────────┐
    │     │             │
"cancel   "terminate    "end subscription
subscription" account"   plan"
    │     │             │
   Retrieve each (with ranks 1..k)
    │     │             │
    └─────┴─────────────┘
          │
   Reciprocal Rank Fusion
          │
    Final ranked results
```

### Reciprocal Rank Fusion

RRF scores each document as the sum of `1 / (rank + k)` across all query variations, where `k=60` is a smoothing constant. Documents that consistently appear at the top across multiple queries get the highest scores.

### Full Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from collections import defaultdict
import json


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Slight temp for variety
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def expand_query(query: str, n_variants: int = 4) -> list[str]:
    """
    Generate N synonym/paraphrase variations of the query.
    Always includes the original query.
    """
    prompt = f"""Generate {n_variants - 1} alternative phrasings of the following 
search query. Each variant should express the same intent using different vocabulary,
synonyms, or phrasing styles. Return ONLY a JSON array of strings.

Original query: "{query}"

Rules:
- Preserve the original intent exactly
- Use different vocabulary (synonyms, domain terms, informal/formal variations)
- Do NOT add new constraints or change the meaning
- Include variations in specificity (more general and more specific)

JSON array of {n_variants - 1} alternatives (NOT including the original):"""

    response = llm.invoke(prompt)

    try:
        variants = json.loads(response.content.strip())
        if not isinstance(variants, list):
            return [query]
        return [query] + variants  # Always include original first
    except json.JSONDecodeError:
        return [query]


def reciprocal_rank_fusion(
    ranked_results: list[list[Document]],
    k: int = 60
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_results: List of ranked document lists (each from a different query)
        k: RRF smoothing constant (60 is standard)
    
    Returns:
        List of (document, score) tuples sorted by descending RRF score
    """
    # Map content hash -> (document, accumulated rrf score)
    scores: dict[int, tuple[Document, float]] = {}

    for result_list in ranked_results:
        for rank, doc in enumerate(result_list):
            doc_id = hash(doc.page_content)
            rrf_score = 1.0 / (rank + k)

            if doc_id in scores:
                scores[doc_id] = (scores[doc_id][0], scores[doc_id][1] + rrf_score)
            else:
                scores[doc_id] = (doc, rrf_score)

    # Sort by descending score
    return sorted(scores.values(), key=lambda x: x[1], reverse=True)


def query_expansion_retrieval(
    query: str,
    n_variants: int = 4,
    k_per_variant: int = 5,
    final_k: int = 6
) -> dict:
    """
    Full query expansion + RRF pipeline.
    """
    # Step 1: Expand query
    variants = expand_query(query, n_variants=n_variants)
    print(f"Query variants ({len(variants)} total):")
    for i, v in enumerate(variants):
        label = "(original)" if i == 0 else f"(variant {i})"
        print(f"  {label}: {v}")

    # Step 2: Retrieve for each variant
    all_ranked_results = []
    for variant in variants:
        docs = vectorstore.similarity_search(variant, k=k_per_variant)
        all_ranked_results.append(docs)

    # Step 3: Merge with RRF
    fused = reciprocal_rank_fusion(all_ranked_results)

    # Take top-k
    top_docs = [doc for doc, score in fused[:final_k]]
    scores = [score for doc, score in fused[:final_k]]

    return {
        "docs": top_docs,
        "rrf_scores": scores,
        "query_variants": variants,
        "total_docs_before_fusion": sum(len(r) for r in all_ranked_results)
    }


# Usage
result = query_expansion_retrieval("how do I cancel my subscription")

print(f"\nRetrieved {len(result['docs'])} documents after fusion")
print(f"(from {result['total_docs_before_fusion']} total before dedup)")

for i, (doc, score) in enumerate(zip(result["docs"], result["rrf_scores"]), 1):
    print(f"\n[{i}] RRF score: {score:.4f}")
    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"    Content: {doc.page_content[:150]}...")
```

---

## 12.6 Multi-Query Retrieval

### LangChain's Built-In Solution

LangChain provides `MultiQueryRetriever`, which automates query expansion with minimal boilerplate. It generates N query variations using an LLM and returns the union of all retrieved documents, deduplicated.

### Complete Implementation

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import logging


# ---- Setup ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---- Create MultiQueryRetriever ----
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Enable logging to see which queries are generated
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# ---- Query ----
query = "How do I cancel my subscription?"
docs = multi_query_retriever.invoke(query)

print(f"Retrieved {len(docs)} unique documents")
for doc in docs:
    print(f"  - {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:100]}...")


# ---- Custom Prompt for MultiQueryRetriever ----
# Override the default prompt to control how variants are generated

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import List


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse LLM output into a list of query strings."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]


custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant helping improve document retrieval.
Generate 4 different versions of the following question to retrieve relevant documents.
Use different vocabulary, synonyms, and phrasings. Output one question per line.

Original question: {question}

4 alternative versions:""",
)

custom_retriever = MultiQueryRetriever(
    retriever=base_retriever,
    llm_chain=custom_prompt | llm | LineListOutputParser(),
)

docs = custom_retriever.invoke("How do I cancel my subscription?")
print(f"\nWith custom prompt: {len(docs)} unique documents")
```

### MultiQueryRetriever vs. Manual Query Expansion

| Aspect | MultiQueryRetriever | Manual Expansion + RRF |
|--------|--------------------|-----------------------|
| Code simplicity | Simple (3 lines) | More verbose |
| Ranking | Union, no ranking | RRF-ranked results |
| Control over prompts | Limited (customizable) | Full control |
| Deduplication | Built-in | Must implement |
| Score transparency | None | RRF scores available |
| Best for | Prototyping, standard cases | Production, when ranking matters |

### Integrating Multi-Query into a Full RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


prompt = ChatPromptTemplate.from_template("""Answer the question using only the context below.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:""")

generation_llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_chain = (
    {
        "context": multi_query_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | generation_llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How do I cancel my subscription?")
print(answer)
```

---

## 12.7 Choosing the Right Technique

### Decision Framework

```
What problem are you solving?
│
├── "My queries use different words than the documents"
│   └── Query Expansion or Multi-Query Retrieval
│       (handles synonyms: "cancel" vs "terminate")
│
├── "Questions are too specific, missing relevant background"
│   └── Step-Back Prompting
│       (adds general context alongside specific context)
│
├── "Questions are complex with multiple distinct parts"
│   └── Query Decomposition
│       (breaks "compare X and Y" into separate retrievals)
│
├── "Questions look nothing like document language"
│   └── HyDE
│       (generates answer-like text for embedding)
│
├── "Retrieval fails intermittently and unpredictably"
│   └── Combine Expansion + HyDE, then add Agentic RAG
│       (see Chapter 7.3 for self-correcting retrieval)
│
└── "All of the above at scale"
    └── Multi-query + RRF + HyDE for hardest queries
        Measure with LangSmith (Chapter 4.1) before adding complexity
```

### Comparison Matrix

| Technique | Best For | Extra LLM Calls | Added Latency | Typical Accuracy Gain | Code Complexity |
|-----------|----------|-----------------|---------------|-----------------------|----------------|
| HyDE | Q&A over formal docs, vocabulary mismatch | 1 | Low (~200ms) | High | Low |
| Query Decomposition | Multi-part, comparison questions | 2-6+ | Medium (~1-2s) | Very High | Medium |
| Step-Back Prompting | Specific questions needing general context | 1 | Low (~200ms) | Medium-High | Low |
| Query Expansion + RRF | Synonym/paraphrase problems | 1 | Low (~300ms) | Medium | Medium |
| MultiQueryRetriever | General improvement, prototyping | 1 | Low (~300ms) | Medium | Very Low |

### Combining Techniques

Techniques are composable. A high-quality production pipeline might chain several:

```python
def production_query_pipeline(query: str) -> dict:
    """
    Combined query transformation pipeline.
    Uses decomposition for complex queries, HyDE for simple ones.
    Always applies query expansion on each sub-question.
    """
    # Step 1: Classify query complexity
    classify_prompt = f"""Is this question simple (single topic) or complex 
(multiple topics or requiring comparison)?
Return ONLY "simple" or "complex".

Question: {query}
Classification:"""

    classification = llm.invoke(classify_prompt).content.strip().lower()

    if "complex" in classification:
        # Use decomposition for complex queries
        sub_questions = decompose_query(query, llm)
    else:
        # Simple query: treat as single sub-question
        sub_questions = [query]

    # Step 2: For each sub-question, use expansion + RRF retrieval
    all_docs = []
    sub_answers = []

    for sub_q in sub_questions:
        retrieval = query_expansion_retrieval(sub_q, n_variants=3, k_per_variant=4, final_k=4)
        docs = retrieval["docs"]
        all_docs.extend(docs)

        if len(sub_questions) > 1:
            # Answer each sub-question independently
            answer = answer_subquestion(sub_q, docs, llm)
            sub_answers.append(answer)

    # Step 3: Generate final answer
    if len(sub_questions) > 1:
        final_answer = synthesize_final_answer(query, sub_questions, sub_answers, llm)
    else:
        # Single question: standard generation
        context = "\n\n".join(doc.page_content for doc in all_docs[:6])
        response = llm.invoke(
            f"Answer using only this context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        final_answer = response.content.strip()

    return {
        "answer": final_answer,
        "sub_questions": sub_questions,
        "query_type": classification,
        "total_docs": len(all_docs)
    }
```

### Implementation Order: Where to Start

Add techniques incrementally, measuring impact at each step:

```
Week 1: Baseline (standard RAG)
        └── Measure retrieval quality with LangSmith

Week 2: Add MultiQueryRetriever
        └── Cheapest gain, almost no new code

Week 3: Add HyDE for question-heavy use cases
        └── High impact for documentation Q&A

Week 4: Add Query Decomposition for complex queries
        └── Detect complexity with a classifier, route accordingly

Week 5+: Add Step-Back for domain-specific knowledge gaps
          └── Profile which queries still fail before adding complexity
```

### Production Checklist

Before deploying any query transformation technique:

- [ ] Measure baseline retrieval quality (LangSmith traces, manual spot-checks)
- [ ] Log all generated sub-questions, hypothetical answers, and query variants
- [ ] Add latency budgets -- transformation should not exceed 2-3 seconds total
- [ ] Implement fallbacks -- if the LLM call for transformation fails, fall back to the original query
- [ ] Set token limits on hypothetical answers and decomposition output to prevent runaway costs
- [ ] A/B test with real traffic before full rollout
- [ ] Monitor cost per query after enabling transformations (extra LLM calls add up at scale)
