# 7. Advanced RAG Techniques (The Cutting Edge)

These are the techniques that separate naive "chunk and pray" RAG from production-grade intelligent systems.

---

## 7.1 Long Context vs. RAG -- When to Use Which

### Cost and Latency Comparison

| Metric | Long Context (100k tokens) | RAG (retrieval-based) |
|--------|---------------------------|----------------------|
| **Cost per query** | ~$0.25 | ~$0.01 |
| **Latency** | ~45 seconds | ~1 second |
| **Cost ratio** | 1,200x more expensive | 1x (baseline) |
| **Latency ratio** | 45x slower | 1x (baseline) |

### Decision Framework

```
Should you use Long Context or RAG?
│
├── Document corpus < 50k tokens?
│   ├── Yes --> Long Context might be fine
│   └── No  --> Use RAG
│
├── Query volume > 100/day?
│   ├── Yes --> RAG (cost savings compound)
│   └── No  --> Either works
│
├── Need precise answers with citations?
│   ├── Yes --> RAG
│   └── No  --> Either works
│
├── Need to analyze the ENTIRE document?
│   ├── Yes --> Long Context
│   └── No  --> RAG
│
└── Cost sensitive?
    ├── Yes --> RAG (no question)
    └── No  --> Long Context for simplicity
```

### Example: Cost at Scale

```python
def compare_costs(queries_per_day: int, days: int = 30):
    """Compare monthly costs between Long Context and RAG."""
    
    total_queries = queries_per_day * days
    
    long_context_cost = total_queries * 0.25    # $0.25 per query
    rag_cost = total_queries * 0.01             # $0.01 per query
    
    print(f"Queries: {total_queries:,} per month")
    print(f"Long Context: ${long_context_cost:,.2f}/month")
    print(f"RAG:          ${rag_cost:,.2f}/month")
    print(f"Savings:      ${long_context_cost - rag_cost:,.2f}/month ({(1 - rag_cost/long_context_cost)*100:.0f}%)")

compare_costs(100)
# Queries: 3,000 per month
# Long Context: $750.00/month
# RAG:          $30.00/month
# Savings:      $720.00/month (96%)

compare_costs(1000)
# Queries: 30,000 per month
# Long Context: $7,500.00/month
# RAG:          $300.00/month
# Savings:      $7,200.00/month (96%)
```

---

## 7.2 Contextual Retrieval (Anthropic's Technique)

### The Problem

Naive chunks lose context. A chunk might say "the company" but which company? Or "as mentioned above" -- but the "above" is in a different chunk.

```
Original Document:
"Acme Corp was founded in 1995. The company specializes in cloud 
infrastructure. Their flagship product, AcmeCloud, handles over 
10 billion requests per day."

After Naive Chunking:
Chunk 1: "Acme Corp was founded in 1995."
Chunk 2: "The company specializes in cloud infrastructure."  <-- WHICH company?
Chunk 3: "Their flagship product, AcmeCloud, handles over 10 billion..."  <-- WHOSE product?
```

### The Solution: LLM-Prepended Context

Before embedding, use an LLM to add context to each chunk that explains **where it comes from** and **what it refers to**.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def add_context_to_chunk(chunk_text: str, full_document: str, doc_title: str) -> str:
    """Use an LLM to prepend context to a chunk before embedding."""
    
    prompt = f"""You are processing a document for a search index.

Full Document Title: {doc_title}

Full Document (for reference):
{full_document[:3000]}  

Chunk to contextualize:
{chunk_text}

Write a 50-word (maximum) context prefix for this chunk. 
Start with "This chunk is from..." and explain:
- Which document and section this is from
- What entities or concepts "the company", "they", "it", etc. refer to
- Any other context needed to understand this chunk in isolation

Context prefix:"""
    
    response = llm.invoke(prompt)
    context = response.content.strip()
    
    # Prepend context to the chunk
    return f"{context}\n\n{chunk_text}"


# ---- Full Pipeline ----
full_document = """
# Acme Corp Annual Report 2024

## Company Overview
Acme Corp was founded in 1995 by Jane Smith. The company specializes 
in cloud infrastructure and employs over 5,000 people worldwide.

## Products
Their flagship product, AcmeCloud, handles over 10 billion requests 
per day. The platform supports multi-region deployment and automatic 
scaling.

## Financial Performance
Revenue grew 34% year-over-year to $2.1 billion. The company 
attributes this growth to enterprise adoption of AcmeCloud.
"""

# Step 1: Chunk the document
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_text(full_document)

# Step 2: Add context to each chunk (done at INDEX time, not query time)
contextualized_chunks = []
for chunk in chunks:
    contextualized = add_context_to_chunk(chunk, full_document, "Acme Corp Annual Report 2024")
    contextualized_chunks.append(contextualized)
    print(f"ORIGINAL:       {chunk[:80]}...")
    print(f"CONTEXTUALIZED: {contextualized[:120]}...")
    print("---")

# Example output:
# ORIGINAL:       "The company specializes in cloud infrastructure and employs..."
# CONTEXTUALIZED: "This chunk is from the Company Overview section of the Acme Corp Annual Report 
#                  2024. 'The company' refers to Acme Corp, founded by Jane Smith.
#
#                  The company specializes in cloud infrastructure and employs..."

# Step 3: Embed the CONTEXTUALIZED chunks (not the originals)
vectors = embeddings.embed_documents(contextualized_chunks)

# Now when someone searches "What does Acme Corp do?", the chunk about
# "The company specializes in cloud infrastructure" will match because
# it now includes "Acme Corp" in its context prefix!
```

### Results

| Technique | Top-20 Retrieval Failure Rate |
|-----------|------------------------------|
| Naive chunking | Baseline |
| Contextual retrieval | **49% fewer failures** |
| Contextual retrieval + re-ranking | **67% fewer failures** |

### Cost Analysis

- Cost: ~1-5 cents per document (one-time at indexing)
- This cost is paid at **index time**, not query time
- For 10,000 documents: ~$50-500 total (one-time)
- The accuracy improvement is massive for the cost

---

## 7.3 Agentic RAG (Self-Correcting Retrieval)

### The Problem

Traditional RAG is **one-shot**. If the retrieval is poor (wrong chunks, bad query), the LLM has no way to recover. It just generates a bad answer from bad context.

### The Solution: A LangGraph State Machine That Self-Corrects

```
                    ┌──────────┐
         start ──> │ retrieve  │
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │  grade   │ (LLM evaluates relevance)
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │  router  │
                    └──┬───┬───┘
                       │   │   
            ┌──────────┘   └──────────┐
            │ High relevance          │ Low relevance
            │                         │ + retries left
       ┌────┴─────┐             ┌─────┴────┐
       │ generate  │             │ rewrite  │ (LLM improves the query)
       └────┬─────┘             └─────┬────┘
            │                         │
           END                   Back to retrieve
                                 (self-correcting loop!)
            
            If no retries left:
       ┌─────────┐
       │ fallback │ (graceful error)
       └────┬────┘
            │
           END
```

### Full Implementation

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


class AgenticRAGState(TypedDict):
    query: str              # Original user query
    current_query: str      # May be rewritten
    documents: list         # Retrieved documents
    relevance_score: float  # Average relevance of retrieved docs
    response: str           # Final answer
    attempt: int            # Current attempt number
    max_attempts: int       # Max retries before fallback


llm = ChatOpenAI(model="gpt-4", temperature=0)
grader_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Cheaper for grading
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    """Search the vector store with the current query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["current_query"])
    return {**state, "documents": docs}


def grade_node(state: AgenticRAGState) -> AgenticRAGState:
    """Use an LLM to evaluate how relevant each document is to the query."""
    scores = []
    
    for doc in state["documents"]:
        prompt = f"""Rate the relevance of this document to the query.
Return ONLY a number between 0 and 1.

Query: {state['current_query']}
Document: {doc.page_content[:500]}

Relevance score (0-1):"""
        
        response = grader_llm.invoke(prompt)
        try:
            score = float(response.content.strip())
            scores.append(min(max(score, 0), 1))  # Clamp to [0, 1]
        except ValueError:
            scores.append(0.5)  # Default if parsing fails
    
    avg_score = sum(scores) / len(scores) if scores else 0
    return {**state, "relevance_score": avg_score}


def router_node(state: AgenticRAGState) -> str:
    """Decide: generate, rewrite, or fallback."""
    if state["relevance_score"] >= 0.7:
        return "generate"           # Good enough, generate answer
    elif state["attempt"] < state["max_attempts"]:
        return "rewrite"            # Try again with a better query
    else:
        return "fallback"           # Give up gracefully


def rewrite_node(state: AgenticRAGState) -> AgenticRAGState:
    """Use an LLM to rewrite the query for better retrieval."""
    prompt = f"""The following search query did not return relevant results.
Rewrite it to be more specific and likely to match relevant documents.

Original query: {state['query']}
Current query: {state['current_query']}
Attempt: {state['attempt'] + 1}

Rewritten query:"""
    
    response = llm.invoke(prompt)
    new_query = response.content.strip()
    
    return {
        **state, 
        "current_query": new_query, 
        "attempt": state["attempt"] + 1
    }


def generate_node(state: AgenticRAGState) -> AgenticRAGState:
    """Generate the final answer from the retrieved context."""
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    
    prompt = f"""Answer based ONLY on the following context.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {state['query']}

Answer:"""
    
    response = llm.invoke(prompt)
    return {**state, "response": response.content}


def fallback_node(state: AgenticRAGState) -> AgenticRAGState:
    """Return a graceful error after all retries are exhausted."""
    return {
        **state, 
        "response": f"I wasn't able to find relevant information about "
                    f"'{state['query']}' after {state['attempt']} attempts. "
                    f"Please try rephrasing your question or contact support."
    }


# ---- Build the Graph ----
graph = StateGraph(AgenticRAGState)

# Add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)
graph.add_node("fallback", fallback_node)

# Set entry point
graph.set_entry_point("retrieve")

# Define edges
graph.add_edge("retrieve", "grade")

graph.add_conditional_edges(
    "grade",
    router_node,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback"
    }
)

# Self-correcting loop: rewrite -> retrieve again
graph.add_edge("rewrite", "retrieve")

# Terminal nodes
graph.add_edge("generate", END)
graph.add_edge("fallback", END)

# Compile
agentic_rag = graph.compile()


# ---- Usage ----
result = agentic_rag.invoke({
    "query": "How do I handle API token expiration?",
    "current_query": "How do I handle API token expiration?",
    "documents": [],
    "relevance_score": 0.0,
    "response": "",
    "attempt": 0,
    "max_attempts": 3
})

print(f"Answer: {result['response']}")
print(f"Attempts: {result['attempt']}")
print(f"Final relevance: {result['relevance_score']:.2f}")
```

### When to Use Agentic RAG

| Scenario | Standard RAG | Agentic RAG |
|----------|-------------|-------------|
| Simple FAQ | Sufficient | Overkill |
| Complex multi-part queries | May fail | Self-corrects |
| High-stakes (legal, medical) | Risky | Safer |
| Cost-sensitive | Cheaper | More expensive (extra LLM calls) |
| Latency-sensitive | Faster | Slower (multiple rounds) |

---

## 7.4 Graph RAG (Microsoft)

### The Problem

Traditional RAG retrieves **isolated chunks**. It cannot answer questions that require connecting facts from different parts of the knowledge base.

**Example**: "Who are the competitors of companies that John advises?"
- Chunk A: "John Smith advises Tech Corp"
- Chunk B: "Tech Corp competes with Data Inc"
- Traditional RAG might find Chunk A but miss Chunk B. It can't "traverse" from John to Tech Corp to Data Inc.

### The Solution: Knowledge Graphs

Build a graph of **entities** (nodes) and **relationships** (edges), then traverse it to answer complex queries.

```
Knowledge Graph:
                                    
  (John Smith) ──[advises]──> (Tech Corp) ──[competes with]──> (Data Inc)
       │                           │                              │
  [works at]                 [founded by]                   [founded by]
       │                           │                              │
  (Advisory LLC)             (Jane Doe)                     (Bob Lee)
                                   │
                             [board member]
                                   │
                              (Global Fund)
```

### Implementation Workflow

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)


# ---- Step 1: Entity Extraction ----
def extract_entities_and_relationships(text: str) -> dict:
    """Use an LLM to extract entities and relationships from text."""
    
    prompt = f"""Extract all entities and relationships from the following text.

Return as JSON with this structure:
{{
    "entities": [
        {{"name": "Entity Name", "type": "PERSON|COMPANY|PRODUCT|LOCATION"}},
    ],
    "relationships": [
        {{"source": "Entity A", "relation": "advises", "target": "Entity B"}},
    ]
}}

Text:
{text}

JSON:"""
    
    response = llm.invoke(prompt)
    import json
    return json.loads(response.content)


# Example
text = """
John Smith is a senior advisor at Advisory LLC. He advises Tech Corp, 
a cloud infrastructure company founded by Jane Doe in 2010. Tech Corp's 
main competitor is Data Inc, which was founded by Bob Lee. Jane Doe 
also serves on the board of Global Fund.
"""

result = extract_entities_and_relationships(text)
print("Entities:", result["entities"])
# [{"name": "John Smith", "type": "PERSON"},
#  {"name": "Advisory LLC", "type": "COMPANY"},
#  {"name": "Tech Corp", "type": "COMPANY"},
#  {"name": "Jane Doe", "type": "PERSON"},
#  {"name": "Data Inc", "type": "COMPANY"},
#  {"name": "Bob Lee", "type": "PERSON"},
#  {"name": "Global Fund", "type": "COMPANY"}]

print("Relationships:", result["relationships"])
# [{"source": "John Smith", "relation": "works_at", "target": "Advisory LLC"},
#  {"source": "John Smith", "relation": "advises", "target": "Tech Corp"},
#  {"source": "Jane Doe", "relation": "founded", "target": "Tech Corp"},
#  {"source": "Tech Corp", "relation": "competes_with", "target": "Data Inc"},
#  {"source": "Bob Lee", "relation": "founded", "target": "Data Inc"},
#  {"source": "Jane Doe", "relation": "board_member", "target": "Global Fund"}]


# ---- Step 2: Build the Graph ----
import networkx as nx

def build_knowledge_graph(extraction: dict) -> nx.DiGraph:
    """Build a NetworkX graph from extracted entities and relationships."""
    G = nx.DiGraph()
    
    # Add nodes
    for entity in extraction["entities"]:
        G.add_node(entity["name"], type=entity["type"])
    
    # Add edges
    for rel in extraction["relationships"]:
        G.add_edge(rel["source"], rel["target"], relation=rel["relation"])
    
    return G

graph = build_knowledge_graph(result)


# ---- Step 3: Query the Graph ----
def local_search(graph: nx.DiGraph, query_entity: str, max_hops: int = 2) -> str:
    """Traverse the graph from a starting entity to find related information."""
    
    if query_entity not in graph:
        return f"Entity '{query_entity}' not found in knowledge graph."
    
    context_parts = []
    visited = set()
    
    def traverse(entity, depth):
        if depth > max_hops or entity in visited:
            return
        visited.add(entity)
        
        # Get outgoing relationships
        for _, target, data in graph.out_edges(entity, data=True):
            context_parts.append(f"{entity} --[{data['relation']}]--> {target}")
            traverse(target, depth + 1)
        
        # Get incoming relationships
        for source, _, data in graph.in_edges(entity, data=True):
            context_parts.append(f"{source} --[{data['relation']}]--> {entity}")
            traverse(source, depth + 1)
    
    traverse(query_entity, 0)
    return "\n".join(context_parts)


# Multi-hop query: "Who are the competitors of companies that John advises?"
context = local_search(graph, "John Smith", max_hops=2)
print("Graph context:")
print(context)
# John Smith --[advises]--> Tech Corp
# John Smith --[works_at]--> Advisory LLC
# Tech Corp --[competes_with]--> Data Inc
# Jane Doe --[founded]--> Tech Corp

# Now feed this context to an LLM
answer_prompt = f"""Based on the following knowledge graph relationships, 
answer the question.

Relationships:
{context}

Question: Who are the competitors of companies that John advises?

Answer:"""

answer = llm.invoke(answer_prompt)
print(f"\nAnswer: {answer.content}")
# "John Smith advises Tech Corp. Tech Corp's competitor is Data Inc."
```

### Local vs Global Search

| Search Type | Use Case | How It Works |
|-------------|----------|-------------|
| **Local** | Specific, multi-hop questions | Start at entity, traverse N hops |
| **Global** | Holistic overview questions | Use community summaries across the graph |

### When to Use Graph RAG

- Dense relationship data (org charts, supply chains)
- Legal documents with cross-references
- Complex queries requiring multi-hop reasoning
- NOT needed for simple Q&A over isolated documents

---

## 7.5 Multi-Modal RAG (ColPali)

### The Problem

Text extraction from PDFs **destroys visual context**:
- Table structures become jumbled text
- Charts and diagrams are completely lost
- Layout and formatting information disappears

### The Solution: Embed Page Images, Not Text

```
Traditional RAG:
PDF ──> Extract Text ──> Chunk Text ──> Embed Text ──> Search
        (loses visuals)

Multi-Modal RAG (ColPali):
PDF ──> Convert to Images ──> Embed Images ──> Search ──> Send Images to Vision LLM
        (preserves everything)
```

### Implementation

```python
# Multi-Modal RAG with ColPali
# Note: This requires specialized libraries and models

from pdf2image import convert_from_path
from PIL import Image
import base64
import io


# ---- Step 1: Convert PDF Pages to Images ----
def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Convert each PDF page to an image."""
    images = convert_from_path(pdf_path, dpi=150)
    return images

images = pdf_to_images("financial_report.pdf")
print(f"Converted {len(images)} pages to images")


# ---- Step 2: Embed Images with ColPali ----
# ColPali embeds images directly -- no text extraction needed
# Pseudocode (actual implementation depends on the ColPali SDK):

# from colpali import ColPaliModel
# model = ColPaliModel.from_pretrained("vidore/colpali-v1.2")
# 
# # Embed all page images
# page_embeddings = []
# for i, image in enumerate(images):
#     embedding = model.embed_image(image)
#     page_embeddings.append({
#         "page": i + 1,
#         "embedding": embedding,
#         "image": image
#     })
# 
# # Embed the query (ColPali embeds text queries too)
# query = "What was the revenue growth in Q3?"
# query_embedding = model.embed_text(query)
# 
# # Search: find most relevant page images
# from numpy import dot
# similarities = [dot(query_embedding, pe["embedding"]) for pe in page_embeddings]
# top_pages = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]


# ---- Step 3: Send Images to Vision LLM ----
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def query_with_images(query: str, images: list[Image.Image]) -> str:
    """Send page images + query to a vision-capable LLM."""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Vision-capable
    
    # Build message with images
    image_messages = []
    for img in images:
        b64 = image_to_base64(img)
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Based on these document pages, answer: {query}"},
                *image_messages
            ]
        }
    ]
    
    response = llm.invoke(messages)
    return response.content


# Usage:
# answer = query_with_images(
#     "What was the revenue growth in Q3?",
#     [images[top_pages[0]], images[top_pages[1]]]  # Top 2 matching pages
# )
# print(answer)
# "According to the financial table on page 7, Q3 revenue grew 34% YoY..."
```

### Trade-offs

| Aspect | Text-Only RAG | Multi-Modal RAG (ColPali) |
|--------|--------------|--------------------------|
| Cost | $0.01/query | ~$0.10/query (10x more) |
| Accuracy on tables | Poor (text extraction mangles tables) | Excellent (sees the actual table) |
| Accuracy on charts | Zero (charts are images) | Excellent |
| Setup complexity | Low | High |
| Best for | Text-heavy docs | Financial reports, technical diagrams, forms |

---

## Summary: When to Use Each Technique

```
Simple Q&A over text docs
└── Standard RAG (Chapter 1-3)

Need better retrieval accuracy
├── Contextual Retrieval (49-67% fewer failures)
└── Parent Document Retriever + Compression

Complex multi-hop questions
└── Graph RAG (knowledge graph traversal)

Queries that need reformulation
└── Agentic RAG (self-correcting loops)

Documents with tables, charts, diagrams
└── Multi-Modal RAG (ColPali + vision LLM)

Large corpus, high volume, cost-sensitive
└── RAG over Long Context (always)
```
