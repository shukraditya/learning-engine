# ATT Backend - Tree Decomposer

Multi-pass LLM pipeline for processing CS textbooks into hierarchical Knowledge Trees.

## Architecture

```
PDF Input
  ↓
Phase 0: Marker → Full book Markdown (saved for review)
  ↓
Phase 1: LLM extracts TOC → Chapter boundaries
  ↓
Phase 1.5: Split Markdown by chapters
  ↓
Phase 2: LLM decomposes each chapter Markdown → Sections/concepts
  ↓
Phase 3: LLM generates exercises for each concept
  ↓
TreeBuilder → Hierarchical nodes with prerequisites
  ↓
Output: chapter-*.json + visualizable knowledge graph
```

**Key improvement**: PDF is first converted to clean Markdown (Phase 0), which is then:
1. Saved for human review/editing
2. Used as structured input for LLM decomposition
3. Split by chapters for targeted processing

## Installation

```bash
cd 01-backend
uv pip install -e ".[dev]"
```

## Usage

### Default: Ollama (Local-First)

```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Pull the model (first time only)
ollama pull llama3:8b

# 3. Ingest a book (shows rich progress bar)
python -m cli.ingest --source "../03-data/vault/books/raw/Database Internals.pdf"
```

The CLI shows a rich progress interface:
- **Phase 0**: PDF → Markdown (Marker)
- **Phase 1**: Book structure extraction (TOC)
- **Phase 1.5**: Split Markdown by chapters
- **Phase 2**: Chapter decomposition → sections/concepts
- **Phase 3**: Exercise generation for each concept

### Alternative: OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python -m cli.ingest \
  --source "../03-data/vault/books/raw/Database Internals.pdf" \
  --provider openai
```

### Test Mode (First 3 Chapters Only)

```bash
python -m cli.ingest \
  --source "book.pdf" \
  --max-chapters 3 \
  --skip-exercises
```

### With Debug Output

```bash
python -m cli.ingest --source "book.pdf" --verbose
```

### Convert PDF to Markdown (Review First)

```bash
# Just convert to Markdown (no LLM processing)
python -m cli.ingest convert "book.pdf"

# Convert and split by chapters
python -m cli.ingest convert "book.pdf" --split-chapters

# Custom output path
python -m cli.ingest convert "book.pdf" -o "my-book.md"
```

This lets you review/edit the Markdown before running the full pipeline.

## Commands

| Command | Description |
|---------|-------------|
| `ingest` | Process PDF into knowledge tree (full pipeline) |
| `convert` | PDF → Markdown only (for review) |
| `validate` | Validate chapter JSON against schema |
| `list-chapters` | List chapters in processed book |
| `visualize` | Launch interactive tree visualizer |

## Visualizer

Interactive D3.js visualization of the knowledge tree:

```bash
# Launch visualizer (auto-opens browser)
python -m cli.ingest visualize

# Or with specific book
python -m cli.ingest visualize "../03-data/vault/books/processed/database-internals"
```

**Features:**
- Click nodes to view details, content, and exercises
- Expand/collapse sections
- Search nodes by name or ID
- Click prerequisites to jump between related concepts
- Color-coded: Purple=Chapter, Cyan=Section, Green=Concept
- Orange dot = has coding exercise

Open `02-frontend/public/tree-visualizer.html` directly in browser to load JSON files manually.

## Output Structure

```
03-data/vault/books/processed/
└── database-internals/
    ├── database-internals.md          # Full book Markdown
    ├── structure.json                  # Book TOC metadata
    ├── chapters/
    │   ├── ch01-introduction.md        # Chapter 1 Markdown
    │   ├── ch02-storage-engines.md     # Chapter 2 Markdown
    │   └── ...
    ├── chapter-001.json                # Chapter 1 tree (JSON)
    ├── chapter-002.json                # Chapter 2 tree (JSON)
    └── ...
```

Each chapter file contains:
- `ChapterManifest`: Metadata + list of `KnowledgeNode`s
- Each node: id, level, breadcrumb, content, metadata, exercise_config

## Key Files

| File | Purpose |
|------|---------|
| `core/node_schema.py` | Pydantic models for tree structures |
| `core/tree_engine.py` | Main orchestrator (3-pass pipeline) |
| `services/llm_factory.py` | LLM provider abstraction |
| `cli/ingest.py` | CLI entry point |
