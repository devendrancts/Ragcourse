# 11. Document Loading and Preprocessing

Document loading and preprocessing is the unglamorous but absolutely critical first stage of any RAG pipeline. The fanciest retrieval algorithm in the world cannot compensate for documents that were loaded incorrectly, never cleaned, or stripped of their metadata.

---

## 11.1 The Importance of Preprocessing

### Garbage In = Garbage Out

This is the most important principle in data engineering, and it applies with full force to RAG systems. Every problem you introduce at the loading stage compounds through every downstream stage:

```
Poor Loading
    → Corrupted or incomplete text
        → Incoherent chunks
            → Misleading embeddings
                → Wrong documents retrieved
                    → Hallucinated or wrong answer
```

There is no retrieval technique, no re-ranking algorithm, and no prompt trick that can recover from fundamentally broken input data. A document loaded with garbled Unicode, scattered page headers mixed into the body text, and no metadata is a liability, not an asset.

### What "Preprocessing" Actually Means

Preprocessing is not a single step. It is a sequence of transformations applied to raw document bytes before any chunking or embedding happens:

| Stage | What It Does | Why It Matters |
|-------|--------------|----------------|
| **Loading** | Reads raw bytes and extracts text | Determines what text you even have |
| **Cleaning** | Removes noise (headers, footers, artifacts) | Prevents garbage from entering chunks |
| **Normalization** | Standardizes encoding, whitespace, casing | Makes text consistent for embeddings |
| **Deduplication** | Removes repeated content | Prevents one topic from dominating retrieval |
| **Metadata enrichment** | Attaches source, page, section info | Enables filtering and citations |

Skipping any of these stages creates a specific, predictable failure mode in your pipeline.

### The Real Cost of Bad Preprocessing

Consider a legal document corpus. If you load PDFs naively:

- **Headers and footers** ("Page 12 of 47 — CONFIDENTIAL — Smith & Associates LLP") get embedded into every chunk, polluting embeddings with boilerplate noise.
- **Page number artifacts** ("- 12 -") interrupt sentences, creating chunks that split mid-thought.
- **Column-layout PDFs** get their columns merged left-to-right across the page, producing word salad.
- **Scanned PDFs without OCR** produce empty strings or garbage characters.

Each of these causes retrieval failures that look like the LLM "doesn't know" the answer — but the real problem is that the document was never properly loaded.

---

## 11.2 Document Loaders by Type

LangChain's `document_loaders` module provides specialized loaders for dozens of file formats. The right loader depends on your file type, content complexity, and performance requirements.

### PDF Documents

PDFs are the most common enterprise document format and the most technically challenging to load correctly. A PDF is not a text file — it is a layout format that encodes character positions on a page. Text extraction is always a reconstruction, not a direct read.

#### PyPDFLoader — The Standard Choice

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("annual_report.pdf")
docs = loader.load()

# Each doc corresponds to one page
print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:500])
print(docs[0].metadata)
# {'source': 'annual_report.pdf', 'page': 0}
```

**When to use it:** Text-heavy PDFs with simple layouts (reports, articles, contracts). Fast, lightweight, and handles most common cases well.

**Limitations:** Struggles with multi-column layouts, tables, and PDFs with embedded images as primary content.

#### PyMuPDFLoader — Faster and More Robust

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("technical_manual.pdf")
docs = loader.load()

# PyMuPDF provides richer metadata automatically
print(docs[0].metadata)
# {
#   'source': 'technical_manual.pdf',
#   'file_path': 'technical_manual.pdf',
#   'page': 0,
#   'total_pages': 84,
#   'format': 'PDF 1.7',
#   'title': 'MX-500 Technical Manual',
#   'author': 'Engineering Dept',
#   'subject': '',
#   'keywords': '',
#   'creator': 'Adobe InDesign',
#   'producer': 'Adobe PDF Library 15.0',
#   'creationDate': "D:20230615120000",
#   'modDate': "D:20230615120000",
#   'trapped': '',
# }
```

**When to use it:** Complex PDFs, large files where performance matters, or when you need the document metadata (author, creation date, title) that PyMuPDF extracts automatically.

**Installation:** `uv add pymupdf`

#### UnstructuredPDFLoader — For Mixed Content

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# "elements" mode preserves document structure
loader = UnstructuredPDFLoader(
    "mixed_content_report.pdf",
    mode="elements",           # "single", "paged", or "elements"
    strategy="hi_res",         # "auto", "fast", "hi_res", or "ocr_only"
)
docs = loader.load()

# Elements mode gives you typed elements
for doc in docs[:5]:
    print(doc.metadata.get("category"), ":", doc.page_content[:100])
# Title : Executive Summary
# NarrativeText : This report covers the financial results for Q3 2024...
# Table : Revenue | Q3 2024 | Q3 2023 | Change...
# Image :
# NarrativeText : Figure 1 shows the revenue breakdown by region...
```

**When to use it:** Documents with tables, images, mixed layouts, or scanned PDFs requiring OCR. The `hi_res` strategy uses a layout detection model for highest accuracy.

**Installation:** `uv add unstructured[pdf]` (heavy dependency — includes layout models)

#### Choosing the Right PDF Loader

```
PDF Content Type          → Recommended Loader
─────────────────────────────────────────────────
Simple text-only          → PyPDFLoader
Complex layout / tables   → PyMuPDFLoader
Scanned / OCR needed      → UnstructuredPDFLoader (strategy="ocr_only")
Mixed: text + images      → UnstructuredPDFLoader (strategy="hi_res")
Maximum speed             → PyMuPDFLoader
```

---

### Web Pages

Web content requires different handling than static files. The main challenges are: HTML noise (navs, ads, cookie banners, scripts), dynamic content loaded via JavaScript, and rate limiting when crawling many pages.

#### WebBaseLoader — Single Pages

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Load a single page, extracting only the article body
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/library/pathlib.html"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("body", "article", "main-content")
        )
    },
)
docs = loader.load()
print(docs[0].page_content[:300])
```

For multiple pages in a single call:

```python
# Load several pages at once
loader = WebBaseLoader(
    web_paths=[
        "https://example.com/docs/getting-started",
        "https://example.com/docs/configuration",
        "https://example.com/docs/api-reference",
    ]
)
# Use lazy_load() to avoid loading all pages into memory at once
for doc in loader.lazy_load():
    print(doc.metadata["source"], "—", len(doc.page_content), "chars")
```

#### RecursiveUrlLoader — Crawling a Site

```python
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
import re

def simple_extractor(html: str) -> str:
    """Strip HTML tags and return plain text."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    # Remove navigation, footer, sidebar
    for tag in soup.find_all(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

loader = RecursiveUrlLoader(
    url="https://docs.mycompany.com/",
    max_depth=3,               # How deep to follow links
    extractor=simple_extractor,
    prevent_outside=True,      # Don't follow links to other domains
    check_response_status=True,
    continue_on_failure=True,  # Don't crash on 404s
    timeout=10,
    headers={"User-Agent": "MyRAGBot/1.0"},
)

docs = loader.load()
print(f"Crawled {len(docs)} pages")
```

#### Cleaning HTML Content

Raw HTML extracted by BeautifulSoup still contains a lot of noise. A proper HTML-to-text transformation handles link URLs, formatting tags, and whitespace:

```python
from langchain_community.document_transformers import Html2TextTransformer

# Convert raw HTML docs to clean markdown-like text
html2text = Html2TextTransformer(ignore_links=False)
clean_docs = html2text.transform_documents(raw_html_docs)
```

For more aggressive cleaning:

```python
import re
from langchain_core.documents import Document

def clean_web_content(doc: Document) -> Document:
    text = doc.page_content

    # Remove URLs that got embedded in text
    text = re.sub(r'https?://\S+', '', text)

    # Remove navigation artifacts (common patterns)
    text = re.sub(r'(Home\s*[>›]\s*){1,5}', '', text)

    # Remove "Last updated: ..." footers
    text = re.sub(r'Last (updated|modified|edited)[^\n]*\n', '', text, flags=re.IGNORECASE)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return Document(page_content=text.strip(), metadata=doc.metadata)
```

---

### CSV and Structured Data

Structured data is a special case. Unlike prose documents, CSV rows are short, self-contained, and have a defined schema.

#### CSVLoader — Basic Loading

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Load entire CSV — each row becomes one Document
loader = CSVLoader(
    file_path="products.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    },
    encoding="utf-8",
)
docs = loader.load()

# Each document contains all column values as text
print(docs[0].page_content)
# product_id: P-1042
# name: Ergonomic Office Chair
# category: Furniture
# price: 349.99
# description: Adjustable lumbar support, breathable mesh back...
```

#### Selecting Columns and Using a Source Column

```python
loader = CSVLoader(
    file_path="knowledge_base.csv",
    source_column="article_id",   # Use this column as the document source in metadata
    content_columns=["title", "body", "tags"],  # Only include these columns
)
docs = loader.load()
print(docs[0].metadata)
# {'source': 'KB-2041', 'row': 0}
```

#### When to Use Structured Data in RAG vs. Direct SQL

This is an important architectural decision:

| Use RAG (embeddings) when... | Use direct SQL/lookup when... |
|------------------------------|-------------------------------|
| Questions are open-ended ("tell me about products similar to X") | Questions are precise ("what is the price of product P-1042?") |
| Data contains prose descriptions | Data is purely numeric or categorical |
| You need semantic similarity ("comfortable chairs") | You need exact matches or ranges ("price < 500") |
| Dataset is small-to-medium (<100k rows) | Dataset is large (millions of rows) |
| Answers require synthesis across rows | Answer is a single row lookup |

For hybrid scenarios, consider a **metadata filter** approach: embed the prose fields, but store numeric/categorical fields as metadata for filtered retrieval.

---

### Code Repositories

Loading source code for RAG requires preserving structure: file paths, language, and logical boundaries (functions, classes) matter more than arbitrary character counts.

#### Loading Source Files

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

# Load all Python files in a repository
loader = DirectoryLoader(
    path="/path/to/repo",
    glob="**/*.py",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    recursive=True,
    show_progress=True,
    use_multithreading=True,
    silent_errors=True,        # Skip binary files that can't be decoded
)
docs = loader.load()

# Each doc's metadata['source'] contains the file path
for doc in docs[:3]:
    print(doc.metadata["source"])
```

#### GitLoader — Loading Directly from a Git Repository

```python
from langchain_community.document_loaders import GitLoader

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="/tmp/langchain_repo",   # Where to clone (or existing clone)
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py"),
)
docs = loader.load()
print(f"Loaded {len(docs)} Python files")
print(docs[0].metadata)
# {
#   'file_path': 'langchain/chains/base.py',
#   'file_name': 'base.py',
#   'file_type': '.py',
#   'repo_id': 'https://github.com/langchain-ai/langchain',
# }
```

#### Respecting .gitignore

```python
import subprocess
from pathlib import Path

def get_gitignored_files(repo_path: str) -> set[str]:
    """Return the set of files that git would ignore."""
    result = subprocess.run(
        ["git", "ls-files", "--ignored", "--exclude-standard", "--others"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    return set(result.stdout.strip().splitlines())

def load_repo_respecting_gitignore(repo_path: str, extensions: list[str]) -> list:
    from langchain_community.document_loaders import TextLoader
    from langchain_core.documents import Document

    ignored = get_gitignored_files(repo_path)
    docs = []

    for ext in extensions:
        for path in Path(repo_path).rglob(f"*{ext}"):
            rel_path = str(path.relative_to(repo_path))
            if rel_path in ignored:
                continue
            if ".git/" in rel_path or "node_modules/" in rel_path:
                continue
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            except Exception:
                pass  # Skip binary or unreadable files

    return docs
```

---

### Other Formats

#### DOCX — Microsoft Word

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("contract.docx")
docs = loader.load()
# Returns a single Document with all text
print(docs[0].page_content[:200])
```

For better structure preservation (tables, headings):

```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

loader = UnstructuredWordDocumentLoader(
    "report.docx",
    mode="elements",
)
docs = loader.load()
```

#### PowerPoint

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader(
    "presentation.pptx",
    mode="elements",
)
docs = loader.load()

# Slides are extracted as Title + NarrativeText elements
for doc in docs:
    print(doc.metadata.get("category"), "|", doc.metadata.get("page_number"), "|", doc.page_content[:80])
```

#### JSON and JSONL

```python
from langchain_community.document_loaders import JSONLoader

# Use a jq-style expression to extract the text field
loader = JSONLoader(
    file_path="knowledge_base.json",
    jq_schema=".[]",           # Iterate over top-level array
    content_key="body",        # Use "body" field as page_content
    metadata_func=lambda record, metadata: {
        **metadata,
        "article_id": record.get("id"),
        "title": record.get("title"),
        "created_at": record.get("created_at"),
    },
)
docs = loader.load()

# JSONL (one JSON object per line)
loader_jsonl = JSONLoader(
    file_path="data.jsonl",
    jq_schema=".",
    content_key="text",
    json_lines=True,
)
```

#### Plain Text

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("readme.txt", encoding="utf-8")
docs = loader.load()

# For a directory of text files
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "/path/to/notes/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"autodetect_encoding": True},  # Handle mixed encodings
)
```

---

## 11.3 Text Cleaning and Normalization

Raw text extracted from documents is never clean. A structured cleaning pipeline applied consistently across all documents is far better than ad-hoc fixes scattered throughout your codebase.

### Common Noise Patterns and Their Fixes

#### 1. Removing Boilerplate

Page headers, footers, and running titles repeat on every page and add nothing to the semantic content of a chunk:

```python
import re

# These patterns appear in legal/financial PDFs
BOILERPLATE_PATTERNS = [
    # Page numbers: "- 12 -", "Page 12", "12 of 47"
    r'[-–]\s*\d+\s*[-–]',
    r'\bPage\s+\d+(\s+of\s+\d+)?\b',
    r'\b\d+\s+of\s+\d+\b',

    # Running headers (company name at top of each page)
    r'^ACME CORPORATION\s*\n',
    r'^CONFIDENTIAL\s*\n',

    # Copyright notices
    r'©\s*\d{4}[^\n]*\n',
    r'Copyright\s+\d{4}[^\n]*\n',

    # "This page intentionally left blank"
    r'[Tt]his page (is )?intentionally left blank\.?',
]

def remove_boilerplate(text: str, patterns: list[str] = BOILERPLATE_PATTERNS) -> str:
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    return text
```

#### 2. Unicode Normalization

Different sources produce different Unicode representations of visually identical characters. NFC normalization collapses these to a canonical form:

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    # NFC: canonical decomposition followed by canonical composition
    # This collapses "e" + combining accent → "é"
    text = unicodedata.normalize("NFC", text)

    # Replace common "smart" punctuation with ASCII equivalents
    replacements = {
        '\u2018': "'",   # Left single quotation mark
        '\u2019': "'",   # Right single quotation mark
        '\u201c': '"',   # Left double quotation mark
        '\u201d': '"',   # Right double quotation mark
        '\u2013': '-',   # En dash
        '\u2014': '--',  # Em dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\ufeff': '',    # BOM
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text
```

#### 3. Whitespace Cleanup

Extracted PDF text is often full of spurious line breaks, double spaces, and irregular indentation:

```python
def normalize_whitespace(text: str) -> str:
    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Collapse multiple spaces to one
    text = re.sub(r' {2,}', ' ', text)

    # Remove spaces at start/end of lines
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)

    # Collapse more than 2 consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix broken sentences: lines that end mid-sentence and continue on next line
    # (Heuristic: if a line doesn't end in punctuation and the next doesn't start
    # with a capital letter, they probably belong together)
    text = re.sub(r'([^.!?\n])\n([a-z])', r'\1 \2', text)

    return text.strip()
```

#### 4. Deduplication

Repeated content degrades retrieval quality by making one topic artificially dominant in your vector space.

```python
import hashlib
from langchain_core.documents import Document

def deduplicate_documents(docs: list[Document]) -> list[Document]:
    """Remove exact-duplicate documents based on content hash."""
    seen_hashes = set()
    unique_docs = []

    for doc in docs:
        # Hash the content (strip whitespace for near-exact matching)
        content_hash = hashlib.md5(
            doc.page_content.strip().encode("utf-8")
        ).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    print(f"Deduplication: {len(docs)} → {len(unique_docs)} docs ({len(docs) - len(unique_docs)} removed)")
    return unique_docs


def near_deduplicate_documents(
    docs: list[Document],
    min_length: int = 50,
) -> list[Document]:
    """Remove near-duplicate documents using shingle hashing (MinHash-style)."""
    from collections import defaultdict

    def shingle_hash(text: str, k: int = 5) -> frozenset:
        """Create a set of character k-gram hashes."""
        text = text.lower().strip()
        shingles = {text[i:i+k] for i in range(len(text) - k + 1)}
        return frozenset(hash(s) for s in shingles)

    def jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    unique_docs = []
    seen_shingles = []

    for doc in docs:
        if len(doc.page_content) < min_length:
            unique_docs.append(doc)
            continue

        shingles = shingle_hash(doc.page_content)
        is_duplicate = any(
            jaccard_similarity(shingles, seen) > 0.85
            for seen in seen_shingles
        )

        if not is_duplicate:
            unique_docs.append(doc)
            seen_shingles.append(shingles)

    return unique_docs
```

### Complete Text Cleaning Pipeline

```python
from langchain_core.documents import Document
from typing import Callable

class TextCleaningPipeline:
    """Composable pipeline for cleaning document text."""

    def __init__(self, steps: list[Callable[[str], str]] | None = None):
        self.steps = steps or [
            normalize_unicode,
            remove_boilerplate,
            normalize_whitespace,
        ]

    def clean_document(self, doc: Document) -> Document:
        text = doc.page_content
        for step in self.steps:
            text = step(text)
        return Document(page_content=text, metadata=doc.metadata)

    def clean_documents(self, docs: list[Document]) -> list[Document]:
        cleaned = [self.clean_document(doc) for doc in docs]
        # Remove documents that are now empty or too short after cleaning
        cleaned = [doc for doc in cleaned if len(doc.page_content.strip()) > 50]
        return cleaned

    def add_step(self, step: Callable[[str], str]) -> "TextCleaningPipeline":
        self.steps.append(step)
        return self


# Usage
pipeline = TextCleaningPipeline()
pipeline.add_step(lambda text: text.lower())  # Optional: lowercase for case-insensitive matching

clean_docs = pipeline.clean_documents(raw_docs)
print(f"Cleaned {len(clean_docs)} documents")
```

---

## 11.4 Metadata Enrichment

Metadata is the part of a Document that does not get embedded — it travels alongside the vector and is used for filtering, attribution, and citation. Rich metadata is the difference between a RAG system that says "I found some information" and one that says "According to Section 4.2 of the Q3 2024 Financial Report (updated June 15, 2024)."

### Why Metadata Matters

**Filtered retrieval:** Instead of searching all 50,000 chunks, filter to only the 300 chunks from this year's policy documents.

**Citations:** Tell the user exactly which document, page, and section the answer came from.

**Debugging:** When retrieval goes wrong, metadata tells you *which* documents are being retrieved, making problems diagnosable.

**Access control:** Attach user permission levels to documents and filter at query time.

### Standard Metadata Fields

```python
from langchain_core.documents import Document
from datetime import datetime
import os

def enrich_metadata(doc: Document, file_path: str) -> Document:
    """Add standard metadata fields to a document."""
    stat = os.stat(file_path)
    path_parts = file_path.replace("\\", "/").split("/")

    metadata = {
        **doc.metadata,  # Preserve loader-provided metadata (page number, etc.)

        # Identity
        "source": file_path,
        "filename": os.path.basename(file_path),
        "file_extension": os.path.splitext(file_path)[1].lower(),

        # Timing
        "file_created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),

        # Size
        "file_size_bytes": stat.st_size,
        "content_length": len(doc.page_content),

        # Pipeline tracking
        "loader_version": "1.0",
    }

    return Document(page_content=doc.page_content, metadata=metadata)
```

### Custom Metadata Extraction

For domain-specific documents, extract structured information directly from the text:

```python
import re
from langchain_core.documents import Document

def extract_legal_metadata(doc: Document) -> Document:
    """Extract metadata specific to legal documents."""
    text = doc.page_content

    # Extract document date
    date_match = re.search(
        r'(?:dated?|effective)\s+(\w+ \d{1,2},?\s+\d{4})',
        text,
        re.IGNORECASE,
    )

    # Extract parties
    parties_match = re.search(
        r'(?:between|by and between)\s+([A-Z][^\n,]+?)\s+(?:and|AND)\s+([A-Z][^\n,]+)',
        text,
    )

    # Extract contract type from title
    doc_type_match = re.search(
        r'^(SERVICE AGREEMENT|EMPLOYMENT CONTRACT|NDA|MASTER SERVICES AGREEMENT)',
        text,
        re.MULTILINE,
    )

    metadata = {
        **doc.metadata,
        "document_date": date_match.group(1) if date_match else None,
        "party_1": parties_match.group(1).strip() if parties_match else None,
        "party_2": parties_match.group(2).strip() if parties_match else None,
        "document_type": doc_type_match.group(1) if doc_type_match else "UNKNOWN",
    }

    return Document(page_content=doc.page_content, metadata=metadata)


def extract_section_title(doc: Document) -> Document:
    """Attempt to identify the section title from the document's first line."""
    lines = doc.page_content.strip().splitlines()
    title = None
    for line in lines[:5]:
        line = line.strip()
        # Consider a line a title if it's short, non-empty, and sentence-like
        if 5 < len(line) < 120 and not line.endswith(('.', ',', ':')):
            title = line
            break

    return Document(
        page_content=doc.page_content,
        metadata={**doc.metadata, "section_title": title},
    )
```

### Using LLMs to Extract Metadata (Semantic Enrichment)

For complex documents where regex is insufficient, use a fast LLM to extract structured metadata:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class DocumentMetadata(BaseModel):
    document_type: str = Field(description="Type of document: policy, procedure, faq, technical, legal, financial, other")
    primary_topic: str = Field(description="The main topic in 5 words or fewer")
    department: str | None = Field(description="Department this document belongs to, if apparent")
    is_current: bool = Field(description="Whether the document appears to be current/active vs. archived/historical")
    summary: str = Field(description="One-sentence summary of the document")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(DocumentMetadata)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document classifier. Extract structured metadata from the document excerpt."),
    ("human", "Document (first 500 chars):\n\n{text}\n\nExtract the metadata."),
])

chain = prompt | structured_llm

def enrich_with_llm(doc: Document) -> Document:
    try:
        extracted: DocumentMetadata = chain.invoke({
            "text": doc.page_content[:500],
        })
        return Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "document_type": extracted.document_type,
                "primary_topic": extracted.primary_topic,
                "department": extracted.department,
                "is_current": extracted.is_current,
                "llm_summary": extracted.summary,
            },
        )
    except Exception as e:
        print(f"Metadata extraction failed for {doc.metadata.get('source')}: {e}")
        return doc
```

### Using Metadata for Filtered Retrieval

Once metadata is attached, you can constrain retrieval at query time:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# Retrieve only from current HR policies
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {
            "department": "HR",
            "is_current": True,
            "document_type": "policy",
        },
    },
)

results = retriever.invoke("What is the vacation accrual policy?")
```

---

## 11.5 Document Loading Pipeline

A production document loading pipeline is more than a script that calls `loader.load()`. It needs error handling, progress tracking, incremental updates, and the ability to process thousands of documents reliably.

### Complete Production Pipeline

```python
import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Generator
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Step 1: Loader Registry
# ─────────────────────────────────────────────

LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}

def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        raise ValueError(f"No loader registered for extension: {ext}")

    # Some loaders require extra kwargs
    if loader_cls == TextLoader:
        return loader_cls(file_path, autodetect_encoding=True)
    return loader_cls(file_path)


# ─────────────────────────────────────────────
# Step 2: Load with Error Handling
# ─────────────────────────────────────────────

def load_file_safely(file_path: str) -> list[Document]:
    """Load a single file, returning [] on failure."""
    try:
        loader = get_loader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {file_path}")
        return docs
    except ValueError as e:
        logger.warning(f"Skipping {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {type(e).__name__}: {e}")
        return []


# ─────────────────────────────────────────────
# Step 3: Cleaning Pipeline
# ─────────────────────────────────────────────

def clean_document(doc: Document) -> Document:
    import unicodedata
    import re

    text = doc.page_content

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\u00a0', ' ').replace('\ufeff', '').replace('\u200b', '')

    # Whitespace normalization
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)

    return Document(page_content=text.strip(), metadata=doc.metadata)


# ─────────────────────────────────────────────
# Step 4: Metadata Enrichment
# ─────────────────────────────────────────────

def enrich_metadata(doc: Document, file_path: str) -> Document:
    stat = os.stat(file_path)
    metadata = {
        **doc.metadata,
        "source": file_path,
        "filename": os.path.basename(file_path),
        "file_extension": Path(file_path).suffix.lower(),
        "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),
        "content_hash": hashlib.md5(doc.page_content.encode()).hexdigest(),
    }
    return Document(page_content=doc.page_content, metadata=metadata)


# ─────────────────────────────────────────────
# Step 5: Incremental Loading State
# ─────────────────────────────────────────────

class IncrementalLoadState:
    """Tracks which files have been processed and their content hashes."""

    def __init__(self, state_file: str = ".rag_load_state.json"):
        self.state_file = state_file
        self.state: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_processing(self, file_path: str) -> bool:
        """Return True if the file is new or has changed since last processing."""
        current_hash = self.get_file_hash(file_path)
        stored = self.state.get(file_path, {})
        return stored.get("hash") != current_hash

    def mark_processed(self, file_path: str, doc_count: int):
        self.state[file_path] = {
            "hash": self.get_file_hash(file_path),
            "processed_at": datetime.utcnow().isoformat(),
            "doc_count": doc_count,
        }


# ─────────────────────────────────────────────
# Step 6: Batch Processing
# ─────────────────────────────────────────────

def discover_files(
    directory: str,
    extensions: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Recursively find all loadable files in a directory."""
    if extensions is None:
        extensions = list(LOADER_REGISTRY.keys())
    if exclude_patterns is None:
        exclude_patterns = [".git", "__pycache__", "node_modules", ".venv"]

    files = []
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        if any(pattern in str(path) for pattern in exclude_patterns):
            continue
        if path.suffix.lower() in extensions:
            files.append(str(path))

    return sorted(files)


def process_documents_in_batches(
    file_paths: list[str],
    batch_size: int = 10,
) -> Generator[list[Document], None, None]:
    """Yield processed document batches, one batch at a time."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i : i + batch_size]
        batch_docs = []

        for file_path in batch_files:
            raw_docs = load_file_safely(file_path)
            for doc in raw_docs:
                doc = clean_document(doc)
                doc = enrich_metadata(doc, file_path)
                if len(doc.page_content.strip()) > 50:
                    batch_docs.append(doc)

        # Chunk the cleaned docs
        chunks = splitter.split_documents(batch_docs)
        logger.info(f"Batch {i // batch_size + 1}: {len(batch_files)} files → {len(chunks)} chunks")
        yield chunks


# ─────────────────────────────────────────────
# Step 7: Main Pipeline Entry Point
# ─────────────────────────────────────────────

def run_pipeline(
    documents_dir: str,
    vectorstore_dir: str = "./chroma_db",
    incremental: bool = True,
    batch_size: int = 20,
):
    """
    Full document loading pipeline:
    discover → filter unchanged → load → clean → enrich → chunk → embed → store
    """
    logger.info(f"Starting pipeline for directory: {documents_dir}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=vectorstore_dir,
    )

    state = IncrementalLoadState() if incremental else None

    # Discover files
    all_files = discover_files(documents_dir)
    logger.info(f"Discovered {len(all_files)} files")

    # Filter to only new/changed files if incremental
    if incremental and state:
        files_to_process = [f for f in all_files if state.needs_processing(f)]
        logger.info(f"Incremental mode: {len(files_to_process)} files need processing")
    else:
        files_to_process = all_files

    if not files_to_process:
        logger.info("No files to process. Pipeline complete.")
        return

    # Process in batches and add to vectorstore
    total_chunks = 0
    for chunk_batch in process_documents_in_batches(files_to_process, batch_size):
        if chunk_batch:
            vectorstore.add_documents(chunk_batch)
            total_chunks += len(chunk_batch)

    # Update state
    if incremental and state:
        for file_path in files_to_process:
            if os.path.exists(file_path):
                state.mark_processed(file_path, doc_count=1)
        state.save()
        logger.info(f"State saved: {len(state.state)} files tracked")

    logger.info(f"Pipeline complete. Added {total_chunks} chunks to vectorstore.")
    return vectorstore


# ─────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    vs = run_pipeline(
        documents_dir="./company_docs",
        vectorstore_dir="./chroma_db",
        incremental=True,
        batch_size=20,
    )

    # Verify
    results = vs.similarity_search("What is the vacation policy?", k=3)
    for r in results:
        print(r.metadata.get("source"), "|", r.metadata.get("page"))
        print(r.page_content[:200])
        print("---")
```

### Error Handling Patterns

Production pipelines encounter corrupt files, encoding errors, password-protected PDFs, and network timeouts. Handle these gracefully rather than crashing:

```python
from dataclasses import dataclass, field
from enum import Enum

class LoadStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclass
class LoadResult:
    file_path: str
    status: LoadStatus
    doc_count: int = 0
    error: str | None = None
    duration_seconds: float = 0.0

def load_with_diagnostics(file_path: str) -> LoadResult:
    import time
    start = time.time()

    try:
        loader = get_loader(file_path)
        docs = loader.load()
        return LoadResult(
            file_path=file_path,
            status=LoadStatus.SUCCESS,
            doc_count=len(docs),
            duration_seconds=time.time() - start,
        )
    except ValueError as e:
        return LoadResult(
            file_path=file_path,
            status=LoadStatus.SKIPPED,
            error=str(e),
            duration_seconds=time.time() - start,
        )
    except MemoryError:
        logger.error(f"Out of memory processing {file_path} — file may be too large")
        return LoadResult(
            file_path=file_path,
            status=LoadStatus.FAILED,
            error="MemoryError: file too large",
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return LoadResult(
            file_path=file_path,
            status=LoadStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
            duration_seconds=time.time() - start,
        )


def run_with_report(file_paths: list[str]) -> dict:
    results = [load_with_diagnostics(fp) for fp in file_paths]

    successes = [r for r in results if r.status == LoadStatus.SUCCESS]
    failures = [r for r in results if r.status == LoadStatus.FAILED]
    skipped = [r for r in results if r.status == LoadStatus.SKIPPED]

    report = {
        "total": len(results),
        "succeeded": len(successes),
        "failed": len(failures),
        "skipped": len(skipped),
        "total_docs": sum(r.doc_count for r in successes),
        "avg_duration": sum(r.duration_seconds for r in results) / len(results) if results else 0,
        "failures": [{"path": r.file_path, "error": r.error} for r in failures],
    }

    logger.info(
        f"Load complete: {report['succeeded']} succeeded, "
        f"{report['failed']} failed, {report['skipped']} skipped. "
        f"{report['total_docs']} documents loaded."
    )

    return report
```

### Incremental Loading Deep Dive

A full reload of a large document corpus is expensive. Incremental loading processes only what has changed since the last run.

The key insight is that you need to track *both* the file hash (to detect changes) *and* the vector IDs of previously inserted chunks (so you can delete the old ones before inserting the new version):

```python
import uuid
from langchain_chroma import Chroma

class IncrementalVectorstoreManager:
    """Manages incremental updates to a vectorstore with delete-and-replace semantics."""

    def __init__(self, vectorstore: Chroma, state_file: str = ".vectorstore_state.json"):
        self.vectorstore = vectorstore
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_update(self, file_path: str) -> bool:
        stored = self.state.get(file_path, {})
        if not stored:
            return True
        current_hash = self._file_hash(file_path)
        return stored.get("hash") != current_hash

    def upsert_file(self, file_path: str, chunks: list[Document]):
        """Delete old chunks for this file and insert new ones."""
        # Delete old chunks if they exist
        if file_path in self.state:
            old_ids = self.state[file_path].get("chunk_ids", [])
            if old_ids:
                self.vectorstore.delete(ids=old_ids)
                logger.info(f"Deleted {len(old_ids)} old chunks for {file_path}")

        # Assign new IDs and insert
        new_ids = [str(uuid.uuid4()) for _ in chunks]
        self.vectorstore.add_documents(chunks, ids=new_ids)

        # Update state
        self.state[file_path] = {
            "hash": self._file_hash(file_path),
            "chunk_ids": new_ids,
            "updated_at": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks),
        }
        self._save_state()
        logger.info(f"Upserted {len(chunks)} chunks for {file_path}")

    def remove_deleted_files(self, current_files: list[str]):
        """Remove chunks for files that no longer exist on disk."""
        tracked_files = list(self.state.keys())
        current_set = set(current_files)

        for tracked_path in tracked_files:
            if tracked_path not in current_set:
                old_ids = self.state[tracked_path].get("chunk_ids", [])
                if old_ids:
                    self.vectorstore.delete(ids=old_ids)
                del self.state[tracked_path]
                logger.info(f"Removed deleted file from vectorstore: {tracked_path}")

        self._save_state()
```

### Putting It All Together — Scheduled Incremental Sync

In production, the pipeline runs on a schedule (cron job, Airflow DAG, etc.):

```python
# sync_documents.py — run this on a schedule (e.g., every hour)

def sync():
    logger.info("Starting incremental document sync")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    manager = IncrementalVectorstoreManager(vectorstore)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Discover all current files
    current_files = discover_files("./company_docs")

    # Remove chunks for deleted files
    manager.remove_deleted_files(current_files)

    # Process new/changed files
    updated = 0
    for file_path in current_files:
        if not manager.needs_update(file_path):
            continue

        raw_docs = load_file_safely(file_path)
        if not raw_docs:
            continue

        cleaned = [enrich_metadata(clean_document(doc), file_path) for doc in raw_docs]
        cleaned = [doc for doc in cleaned if len(doc.page_content.strip()) > 50]
        chunks = splitter.split_documents(cleaned)

        manager.upsert_file(file_path, chunks)
        updated += 1

    logger.info(f"Sync complete. {updated} files updated out of {len(current_files)} total.")


if __name__ == "__main__":
    sync()
```

---

## Summary

| Stage | Key Decisions | Common Mistakes |
|-------|---------------|-----------------|
| **Loading** | Match loader to file type and content complexity | Using PyPDFLoader on scanned PDFs (empty output) |
| **Cleaning** | Build a reusable pipeline, not ad-hoc fixes | Skipping cleaning entirely — boilerplate contaminates embeddings |
| **Normalization** | Always apply Unicode NFC and whitespace normalization | Inconsistent character sets breaking semantic similarity |
| **Deduplication** | Hash-based for exact, shingle-based for near-duplicate | Allowing duplicates to skew retrieval toward common content |
| **Metadata** | Attach source, page, date, section at minimum | No metadata — impossible to cite sources or filter results |
| **Incremental loading** | Track file hashes and chunk IDs for delete-and-replace | Full reloads on every run — too slow and costly at scale |

The document loading stage is not exciting, but it is foundational. Every hour spent building a robust, well-instrumented loading pipeline pays dividends across the entire lifecycle of your RAG system.
