# 5. Vector Databases, Scaling, and Cost Optimization

This section covers how to tune, scale, and optimize the cost of your vector database -- the backbone of any RAG system.

---

## 5.1 Vector Database Tuning (HNSW Parameters)

Most vector databases (Chroma, Pinecone, Weaviate, Qdrant) use the **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor search. Two key parameters control its behavior.

### The Two Knobs: M and EF

```
HNSW Graph Visualization:

  M = Max Connections per node         EF = Search Effort (candidate list size)
  
  Low M (8):        High M (32):       Low EF (32):      High EF (200):
  Few connections   Many connections   Quick scan         Deep scan
  
    A --- B           A --- B           Search checks     Search checks
    |                 |\ /|            32 candidates      200 candidates
    C --- D           C-X--D           Fast but may       Slow but finds
                      |/ \|            miss best match    the best match
                      E --- F
```

### Parameter Details

| Parameter | What It Controls | Low Value | High Value |
|-----------|-----------------|-----------|------------|
| **M** (Max Connections) | Number of edges per node in the graph | 8-16: Smaller index, faster build, lower accuracy | 32-64: Larger index, slower build, higher accuracy |
| **EF** (Search Effort) | Size of dynamic candidate list during search | 32-64: Faster search, lower accuracy | 200-500: Slower search, higher accuracy |

### The Fundamental Trade-off

> **You cannot have all three: high accuracy, high speed, small index. Pick two.**

```
                    HIGH ACCURACY
                         /\
                        /  \
                       /    \
                      /      \
        SMALL INDEX /________\ HIGH SPEED
        
        Pick any two. The third suffers.
```

### Recommended Configurations

```python
import chromadb

# ---- Prototype / Development ----
# Priority: Speed for fast iteration
client = chromadb.Client()
collection = client.create_collection(
    name="prototype",
    metadata={
        "hnsw:M": 16,         # Moderate connections
        "hnsw:efConstruction": 40,  # Fast build
        "hnsw:efSearch": 40         # Fast search
    }
)

# ---- Production (Balanced) ----
# Priority: Good accuracy + reasonable speed
collection = client.create_collection(
    name="production_balanced",
    metadata={
        "hnsw:M": 16,
        "hnsw:efConstruction": 100,
        "hnsw:efSearch": 100
    }
)

# ---- Production (High Accuracy) ----
# Priority: Best possible retrieval quality
collection = client.create_collection(
    name="production_accuracy",
    metadata={
        "hnsw:M": 32,              # More connections = better graph
        "hnsw:efConstruction": 200, # Thorough index building
        "hnsw:efSearch": 200        # Thorough search
    }
)
```

### Quick Reference

| Setting | M | EF (Construction) | EF (Search) | Use Case |
|---------|---|-------------------|-------------|----------|
| Prototype | 16 | 40 | 40 | Fast iteration |
| Production Balanced | 16 | 100 | 100 | Most applications |
| Production Accuracy | 32 | 200 | 200 | High-stakes (legal, medical) |

---

## 5.2 Scaling Strategies

### Vertical Scaling (Scale Up)

Get a bigger machine: more RAM, more CPU, faster SSD.

```
┌─────────────────────────────┐
│     Single Large Machine    │
│                             │
│  RAM:  16GB  -->  128GB     │
│  CPU:  4 cores --> 32 cores │
│  SSD:  500GB --> 2TB NVMe   │
│                             │
│  Vectors: up to 5-10M       │
└─────────────────────────────┘

Pros:
  + Simple -- no distributed systems complexity
  + No network latency between nodes
  + Easy to maintain and debug

Cons:
  - Has a ceiling (can't infinitely scale one machine)
  - Single point of failure
  - Expensive at the top end
```

**Best for**: Up to 5-10 million vectors. This is the **easiest first step** and handles most use cases.

### Horizontal Scaling (Sharding)

Split data across multiple machines.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Shard 1    │  │   Shard 2    │  │   Shard 3    │
│              │  │              │  │              │
│  Vectors:    │  │  Vectors:    │  │  Vectors:    │
│  0 - 33M     │  │  33M - 66M   │  │  66M - 100M  │
│              │  │              │  │              │
│  Machine A   │  │  Machine B   │  │  Machine C   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┘                 │
                └──────────┬───────────────┘
                           │
                    ┌──────┴──────┐
                    │  Query      │
                    │  Router     │
                    │  (merges    │
                    │  results)   │
                    └─────────────┘

Pros:
  + Unlimited scale
  + No single point of failure (with replication)
  + Can add capacity incrementally

Cons:
  - Complex to set up and maintain
  - Network latency between shards
  - Requires query routing logic
  - More expensive in operational overhead
```

**Best for**: Over 10 million vectors. Necessary for large-scale enterprise applications.

### Decision Framework

```
How many vectors do you have?
│
├── < 1 million
│   └── Single machine, smallest instance possible
│       ChromaDB or PGVector on a standard VPS
│
├── 1-10 million
│   └── Vertical scaling: bigger machine
│       PGVector on a dedicated database server
│
├── 10-50 million
│   └── Managed service OR start sharding
│       Pinecone, Qdrant Cloud, or self-hosted with sharding
│
└── 50+ million
    └── Horizontal scaling with sharding + replication
        Dedicated vector database cluster
```

---

## 5.3 Vector DB Cost Comparison (at 50 Million Vectors)

| Solution | Monthly Cost | Ops Burden | Control | Best For |
|----------|-------------|------------|---------|----------|
| **Pinecone (Managed)** | ~$1,500+/mo | Zero | Limited | Teams with no DevOps |
| **PGVector (Managed)** | ~$400/mo | Low | Moderate | Teams already using PostgreSQL |
| **PGVector (Self-Hosted)** | ~$300/mo | Significant | Full | Teams with strong DevOps |

### Cost Breakdown

```python
# Quick cost estimation function
def estimate_monthly_cost(
    num_vectors: int,
    dimensions: int = 1536,
    queries_per_day: int = 10000,
    provider: str = "pgvector_managed"
):
    """Rough monthly cost estimate for a vector database."""
    
    # Storage: 4 bytes per float32 dimension
    storage_gb = (num_vectors * dimensions * 4) / (1024**3)
    
    costs = {
        "pinecone": {
            "base": 70,          # Starter pod
            "per_million": 25,    # Per million vectors
            "compute": storage_gb * 5,
        },
        "pgvector_managed": {
            "base": 50,           # Managed PostgreSQL
            "storage": storage_gb * 0.10,  # Per GB
            "compute": 200,       # Dedicated instance
        },
        "pgvector_self_hosted": {
            "base": 0,
            "storage": storage_gb * 0.08,
            "compute": 150,       # VPS cost
            "ops_labor": 100,     # Your time has value!
        }
    }
    
    cost = costs.get(provider, costs["pgvector_managed"])
    total = sum(cost.values())
    
    return {
        "provider": provider,
        "storage_gb": f"{storage_gb:.1f} GB",
        "estimated_monthly": f"${total:.0f}",
        "breakdown": cost
    }

# Compare providers
for provider in ["pinecone", "pgvector_managed", "pgvector_self_hosted"]:
    result = estimate_monthly_cost(50_000_000, provider=provider)
    print(f"{result['provider']}: {result['estimated_monthly']}")
```

---

## 5.4 Cost Optimization Strategies

### Strategy 1: Reduce Embedding Dimensions

```python
from langchain_openai import OpenAIEmbeddings

# Default: 1536 dimensions
default_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Storage per vector: 1536 * 4 bytes = 6,144 bytes

# Optimized: 512 dimensions (saves 66% storage)
optimized_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Reduced!
)
# Storage per vector: 512 * 4 bytes = 2,048 bytes

# At 50M vectors:
# Default:   50M * 6,144 bytes = 307 GB
# Optimized: 50M * 2,048 bytes = 102 GB
# Savings:   205 GB (66% less storage!)
```

### Strategy 2: Quantization

Convert vectors from `float32` to `int8` to save 50-75% on storage.

```python
import numpy as np

# Original vector (float32) -- 4 bytes per dimension
original_vector = np.array([0.234, -0.567, 0.891, ...], dtype=np.float32)
# Size: 1536 dims * 4 bytes = 6,144 bytes

# Quantized vector (int8) -- 1 byte per dimension
def quantize_vector(vector: np.ndarray) -> np.ndarray:
    """Quantize float32 vector to int8 (75% storage reduction)."""
    # Scale to int8 range (-128 to 127)
    min_val = vector.min()
    max_val = vector.max()
    scale = (max_val - min_val) / 255
    quantized = ((vector - min_val) / scale - 128).astype(np.int8)
    return quantized, min_val, scale

quantized, min_val, scale = quantize_vector(original_vector)
# Size: 1536 dims * 1 byte = 1,536 bytes (75% reduction!)

# Dequantize when needed
def dequantize_vector(quantized: np.ndarray, min_val: float, scale: float):
    return (quantized.astype(np.float32) + 128) * scale + min_val
```

### Strategy 3: Response Caching

Cache frequent queries to avoid repeated embedding + LLM calls.

```python
import hashlib
import json
import time
from typing import Optional


class ResponseCache:
    """In-memory cache with TTL for RAG responses."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent cache keys."""
        return query.lower().strip()
    
    def _make_key(self, query: str) -> str:
        """Create SHA-256 hash key from normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Look up a cached response. Returns None if not found or expired."""
        key = self._make_key(query)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["response"]  # Cache HIT
            else:
                del self.cache[key]  # Expired, remove it
        return None  # Cache MISS
    
    def set(self, query: str, response: str):
        """Store a response in the cache."""
        key = self._make_key(query)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query
        }
    
    def stats(self) -> dict:
        """Get cache statistics."""
        valid = sum(1 for e in self.cache.values() 
                    if time.time() - e["timestamp"] < self.ttl)
        return {"total_entries": len(self.cache), "valid_entries": valid}


# ---- Usage in RAG pipeline ----
cache = ResponseCache(ttl_seconds=300)  # 5-minute cache

def cached_rag_query(query: str) -> dict:
    # Step 1: Check cache
    cached = cache.get(query)
    if cached:
        return {"response": cached, "cached": True, "cost": 0.0}
    
    # Step 2: Cache miss -- run full pipeline
    response = rag_chain.invoke(query)
    
    # Step 3: Store in cache for future hits
    cache.set(query, response)
    
    return {"response": response, "cached": False, "cost": 0.012}


# First call: cache miss, full pipeline runs ($0.012)
result1 = cached_rag_query("What is the refund policy?")
print(f"Cached: {result1['cached']}, Cost: ${result1['cost']}")

# Second call (within 5 min): cache hit, instant response ($0.00)
result2 = cached_rag_query("What is the refund policy?")
print(f"Cached: {result2['cached']}, Cost: ${result2['cost']}")

# Slightly different query: also cache hit (normalized)
result3 = cached_rag_query("  What is the Refund Policy?  ")
print(f"Cached: {result3['cached']}, Cost: ${result3['cost']}")
```

### Strategy 4: Batch Embedding Calls

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# BAD: One API call per document (slow, expensive overhead)
vectors_bad = []
for doc in documents:
    vec = embeddings.embed_query(doc.page_content)  # 1 API call each!
    vectors_bad.append(vec)

# GOOD: Batch all documents in one API call
texts = [doc.page_content for doc in documents]
vectors_good = embeddings.embed_documents(texts)  # 1 API call total!
```

### Cost Optimization Summary

| Strategy | Savings | Effort | Impact |
|----------|---------|--------|--------|
| Reduce dimensions (1536->512) | 30-66% storage | Low | Medium |
| Quantization (float32->int8) | 50-75% storage | Medium | Medium |
| Response caching | 50-80% API costs | Low | High |
| Batch embedding calls | 20-40% API overhead | Low | Medium |
| Right-sizing instances | 20-50% compute | Low | Medium |

### Right-Sizing Decision

```
Start small, scale up when monitoring shows a need:

Month 1:  Free tier / smallest instance
          Monitor: latency, error rate, query volume
          
Month 2:  If p99 latency > 500ms --> upgrade instance
          If error rate > 1%     --> check index params
          If queries > 10k/day   --> add caching
          
Month 3+: If vectors > 5M       --> consider managed service
          If vectors > 10M      --> plan sharding strategy
```
