# 2. The 4 Critical Chunking Variables

Chunking is the **single biggest lever** for retrieval quality in a RAG system. Bad chunks = bad embeddings = bad retrieval = bad answers. No amount of prompt engineering can fix poorly chunked data.

---

## 2.1 Variable #1: Chunk Size

### What It Is

Chunk size is the target length of each text fragment, measured in **characters** or **tokens**.

### The Problem: Too Small vs. Too Large

```
TOO SMALL (< 200 tokens)          TOO LARGE (> 1,000 tokens)
┌─────────────────────┐            ┌──────────────────────────────────┐
│ "client ID"         │            │ Page 1: Introduction to APIs...  │
│                     │            │ Page 2: Authentication methods...│
│ No context!         │            │ Page 3: Rate limiting...         │
│ What client?        │            │ Page 4: Error handling...        │
│ What ID format?     │            │ ...10 pages of mixed content     │
│                     │            │                                  │
│ Embedding captures  │            │ Embedding is VAGUE and DILUTED   │
│ an incomplete       │            │ A query for "rate limiting"      │
│ thought             │            │ won't match well because the     │
│                     │            │ vector represents everything     │
└─────────────────────┘            └──────────────────────────────────┘
```

### The Sweet Spot: 200-1,000 Tokens

| Chunk Size | Use Case | Trade-off |
|------------|----------|-----------|
| 200-300 tokens | FAQ, short answers | High precision, low context |
| 400-600 tokens | General documents | Balanced (recommended default) |
| 800-1,000 tokens | Technical docs, legal | More context, less precision |

### Example: Impact of Chunk Size on Retrieval

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

sample_text = """
The API key expires after 24 hours. You must refresh it using the 
/auth/refresh endpoint with your client credentials. The refresh token 
itself has a 30-day expiration. After that, the user must re-authenticate 
completely using the OAuth2 authorization code flow. Rate limiting applies 
to all authentication endpoints: 10 requests per minute per IP address.
"""

# Too small - loses context
small_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
small_chunks = small_splitter.split_text(sample_text)
print("--- TOO SMALL ---")
for i, chunk in enumerate(small_chunks):
    print(f"  Chunk {i}: '{chunk}'")
# Output:
#   Chunk 0: 'The API key expires after 24 hours. You'
#   Chunk 1: 'must refresh it using the'
#   Chunk 2: '/auth/refresh endpoint with your client'
#   ...fragments with no complete meaning

# Just right - preserves complete thoughts
good_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
good_chunks = good_splitter.split_text(sample_text)
print("\n--- JUST RIGHT ---")
for i, chunk in enumerate(good_chunks):
    print(f"  Chunk {i}: '{chunk}'")
# Output:
#   Chunk 0: 'The API key expires after 24 hours. You must refresh it 
#             using the /auth/refresh endpoint with your client credentials.
#             The refresh token itself has a 30-day expiration.'
#   Chunk 1: 'After that, the user must re-authenticate completely using 
#             the OAuth2 authorization code flow. Rate limiting applies to 
#             all authentication endpoints: 10 requests per minute per IP.'
```

---

## 2.2 Variable #2: Overlap

### What It Is

Overlap is the number of characters/tokens **repeated** from the end of one chunk to the beginning of the next.

### The Problem Without Overlap

```
WITHOUT OVERLAP:
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ Chunk 1:                    │  │ Chunk 2:                    │
│ "The API key expires after  │  │ "You must refresh it using  │
│  24 hours."                 │  │  the token endpoint."       │
│                             │  │                             │
│ Has "expiration" keyword    │  │ Has the SOLUTION            │
│ but NOT the solution        │  │ but NOT the keyword         │
└─────────────────────────────┘  └─────────────────────────────┘

Query: "How do I handle API key expiration?"
Result: Matches Chunk 1 (has "expiration"), but Chunk 1 has NO solution!
        Chunk 2 has the solution but is NEVER retrieved!
```

### The Fix With Overlap

```
WITH OVERLAP (50 chars):
┌─────────────────────────────┐  ┌─────────────────────────────────────┐
│ Chunk 1:                    │  │ Chunk 2:                            │
│ "The API key expires after  │  │ "The API key expires after 24       │
│  24 hours."                 │  │  hours. You must refresh it using   │
│                             │  │  the token endpoint."               │
└─────────────────────────────┘  └─────────────────────────────────────┘

Query: "How do I handle API key expiration?"
Result: Matches BOTH chunks! Chunk 2 has the complete context + solution!
```

### Overlap is "Cheap Insurance"

- It introduces a small amount of redundancy (slightly more storage)
- But it **prevents** the frustrating case where the answer exists in your database but cannot be found because it was split across a chunk boundary

### Code Example

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Section A: Authentication
The API key expires after 24 hours. You must refresh it using the token endpoint.
Failure to refresh will result in a 401 Unauthorized error.

Section B: Rate Limiting
All endpoints are rate-limited to 100 requests per minute.
Exceeding the limit returns a 429 Too Many Requests error."""

# Without overlap
no_overlap = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
chunks_no_overlap = no_overlap.split_text(text)

print("--- NO OVERLAP ---")
for i, chunk in enumerate(chunks_no_overlap):
    print(f"Chunk {i}: {chunk}\n")

# With overlap
with_overlap = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
chunks_with_overlap = with_overlap.split_text(text)

print("--- WITH 50-CHAR OVERLAP ---")
for i, chunk in enumerate(chunks_with_overlap):
    print(f"Chunk {i}: {chunk}\n")
```

---

## 2.3 Variable #3: Split Boundaries (Chunking Strategy)

### The 4 Strategies Ranked

### Strategy 1: Fixed-Size Chunking -- AVOID IN PRODUCTION

```python
from langchain.text_splitter import CharacterTextSplitter

# Splits every N characters -- no regard for meaning
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""  # Split on nothing -- pure character count
)

text = "Machine learning is a subset of artificial intelligence. It allows systems to learn from data."
chunks = splitter.split_text(text)
# Could produce: "Machine learning is a subset of artificial intelligen"
#                "ce. It allows systems to learn from data."
# "intelligen" / "ce" -- destroyed a word!
```

**Why it's bad**: Cuts mid-word, mid-sentence. Destroys meaning completely.

### Strategy 2: Recursive Character Text Splitter -- THE DEFAULT

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Splits hierarchically: paragraphs > sentences > words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]  # Priority order
)

text = """
## Authentication

The API uses OAuth 2.0 for authentication. All requests must include a 
Bearer token in the Authorization header.

Tokens expire after 24 hours. Use the refresh endpoint to obtain a new token.

## Rate Limiting

All API endpoints enforce rate limiting. The default limit is 100 requests 
per minute per API key. Enterprise plans have a limit of 1000 requests per minute.

Exceeding the rate limit returns a 429 status code with a Retry-After header.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}\n{'='*50}")

# Result: Splits at paragraph boundaries (\n\n) first
# Chunk 0: "## Authentication\n\nThe API uses OAuth 2.0..."
# Chunk 1: "## Rate Limiting\n\nAll API endpoints enforce..."
```

**How the separator hierarchy works**:
1. First tries to split at `\n\n` (paragraph breaks)
2. If a paragraph is too large, splits at `\n` (line breaks)
3. If a line is too large, splits at `.` (sentences)
4. If a sentence is too large, splits at ` ` (words)
5. Last resort: splits at `""` (characters)

### Strategy 3: Semantic Chunking -- BEST FOR QUALITY

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Splits based on MEANING, not character count
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # How to detect topic shifts
    breakpoint_threshold_amount=90           # 90th percentile = significant shift
)

text = """
Machine learning models require large datasets for training. The quality 
of data directly impacts model performance. Data preprocessing steps include 
normalization, handling missing values, and feature encoding.

The deployment of ML models involves containerization with Docker, 
orchestration with Kubernetes, and monitoring with Prometheus. CI/CD 
pipelines automate the deployment process.

Financial regulations require all trading algorithms to maintain audit 
trails. The SEC mandates real-time reporting of trades exceeding $1 million.
"""

chunks = semantic_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}\n{'='*50}")

# Result: Splits where the TOPIC changes
# Chunk 0: "Machine learning models require... feature encoding."  (ML training topic)
# Chunk 1: "The deployment of ML models... deployment process."    (DevOps topic)
# Chunk 2: "Financial regulations require... exceeding $1M."       (Finance topic)
```

**How it works internally**:
1. Embeds each sentence individually
2. Calculates cosine similarity between adjacent sentences
3. When similarity drops below the threshold (topic shift), it cuts there

### Strategy 4: Late Chunking -- MOST ADVANCED

```python
# Late Chunking embeds the ENTIRE document first, then splits the embeddings
# This means each chunk's embedding carries context from the whole document

# Requires specialized models like jina-embeddings-v2
# Pseudocode (actual implementation depends on the model provider):

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

# Step 1: Embed the ENTIRE document
full_doc = "The entire document text goes here..."
inputs = tokenizer(full_doc, return_tensors="pt")
full_embeddings = model(**inputs).last_hidden_state  # Full document embedding

# Step 2: Split the embeddings (not the text) into chunks
# Each chunk embedding retains context from the full document
chunk_boundaries = [(0, 512), (512, 1024), (1024, 1536)]
chunk_embeddings = [full_embeddings[:, start:end, :].mean(dim=1) 
                    for start, end in chunk_boundaries]

# Result: 10-12% accuracy improvement over standard chunking
# because pronouns and references ("it", "the company") retain their meaning
```

### Strategy Comparison Table

| Strategy | Quality | Speed | Cost | Best For |
|----------|---------|-------|------|----------|
| Fixed-Size | Poor | Fast | Free | Never use in production |
| Recursive | Good | Fast | Free | General text (default choice) |
| Semantic | Excellent | Slow | $$$ (embedding calls) | High-quality unstructured text |
| Late Chunking | Best | Slow | $$ (specialized models) | When accuracy is paramount |

---

## 2.4 Variable #4: Content Type

Different content types need different splitters. Using a general text splitter on code will cut in the middle of a function, making the chunk useless.

### General Text

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### Markdown Documents

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Splits by headers -- each chunk is a logical section
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

markdown_doc = """
# User Guide

## Installation
Run `pip install mypackage` to install. Requires Python 3.8+.

## Configuration
Create a `.env` file with your API key.

### Database Setup
Set DATABASE_URL to your PostgreSQL connection string.

## Usage
Import the package and call `mypackage.run()`.
"""

chunks = markdown_splitter.split_text(markdown_doc)
for chunk in chunks:
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("---")

# Output:
# Content: Run `pip install mypackage`... Requires Python 3.8+.
# Metadata: {'Header 1': 'User Guide', 'Header 2': 'Installation'}
# ---
# Content: Create a `.env` file with your API key.
# Metadata: {'Header 1': 'User Guide', 'Header 2': 'Configuration'}
# ---
# Content: Set DATABASE_URL to your PostgreSQL connection string.
# Metadata: {'Header 1': 'User Guide', 'Header 2': 'Configuration', 'Header 3': 'Database Setup'}
```

### Source Code

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Splits at function/class boundaries
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

python_code = """
class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: str):
        \"\"\"Fetch a user by ID.\"\"\"
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
    
    def create_user(self, name: str, email: str):
        \"\"\"Create a new user.\"\"\"
        user = User(name=name, email=email)
        self.db.add(user)
        self.db.commit()
        return user

class OrderService:
    def __init__(self, db, user_service):
        self.db = db
        self.user_service = user_service
    
    def place_order(self, user_id: str, items: list):
        user = self.user_service.get_user(user_id)
        order = Order(user=user, items=items)
        self.db.add(order)
        self.db.commit()
        return order
"""

chunks = python_splitter.split_text(python_code)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}\n{'='*50}")

# Result: Splits at class/function boundaries
# Chunk 0: class UserService + get_user + create_user
# Chunk 1: class OrderService + place_order
```

### Technical/Legal Documents

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# For legal/technical docs, semantic chunking ensures each chunk
# is a coherent, self-contained concept
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)
```

---

## Quick Reference: Chunking Decision Flowchart

```
What type of content are you chunking?
│
├── General text (articles, docs, reports)
│   └── Use RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
│
├── Markdown documentation
│   └── Use MarkdownHeaderTextSplitter (splits by headers)
│
├── Source code
│   └── Use RecursiveCharacterTextSplitter.from_language()
│
├── High-quality unstructured text (technical/legal)
│   └── Use SemanticChunker (splits by meaning)
│
└── Documents requiring cross-chunk references
    └── Use Late Chunking (embed full doc first, then split)

ALWAYS add overlap (50-100 chars) as "cheap insurance"
ALWAYS target 200-1,000 tokens per chunk
```
