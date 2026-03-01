# Project: Active-Textbook-Tutor (ATT) 
[ATT-Workspace]|root: ./
|IMPORTANT: Adhere to "Hierarchical Discovery" — Logic must flow from Tree Traversal, not Flat Vector Search.
|01-backend:[FastAPI, Pydantic AI, Docling, Python 3.12+]
|  |api:{main.py, dependencies.py, routes/tree.py, routes/agent.py, routes/workspace.py}
|  |core:{tree_engine.py, node_schema.py, markdown_parser.py, metadata_extractor.py}
|  |agents:{
|  |  navigator.py: "The Guide" - Maps user progress to tree nodes.
|  |  architect.py: "The Taskmaster" - Generates code skeletons/tests.
|  |  evaluator.py: "The Socratic Tutor" - Analyzes code vs. Book Theory.
|  |  researcher.py: "The Librarian" - Fetches deep context from sibling/parent nodes.
|  |}
|  |services:{llm_factory.py, code_executor.py, state_manager.py}
|02-frontend:[Vite, React, TypeScript, TailwindCSS]
|  |src/components/layout:{SplitPane.tsx, Sidebar.tsx, Header.tsx}
|  |src/components/tree:{KnowledgeTree.tsx, TreeNode.tsx, ProgressNode.tsx}
|  |src/components/editor:{MonacoWrapper.tsx, Terminal.tsx, DiffView.tsx}
|  |src/components/chat:{SocraticHint.tsx, ThoughtStream.tsx, KeyManager.tsx}
|  |src/store:{useTreeStore.ts, useEditorStore.ts, useAuthStore.ts}
|03-data/vault:[Local Storage Only]
|  |books/raw:{*.pdf, *.epub}
|  |books/processed:{*.json, *.yaml}
|  |users/sessions:{progress.db, telemetry.json}

## 🎯 High-Level Vision
Build a local, privacy-first learning environment that turns static CS textbooks (DDIA, Database Internals, OS/Three Easy Pieces) into interactive coding playgrounds. The system must treat the book as a **Knowledge Tree**, where understanding is validated through incremental code exercises.

## 🏗️ Technical Architecture Standards

### 1. The Knowledge Node (The "Unit of Truth")
Every node in the `processed_trees/*.json` MUST follow this verbose schema:
- `id`: Unique slug (e.g., `storage-engine-lsm-compaction`).
- `level`: Depth (0: Root, 1: Chapter, 2: Section, 3: Concept/Sub-section).
- `breadcrumb`: Full path (e.g., `Storage > Hash Indexes > Bitcask`).
- `content`: Cleaned Markdown with LaTeX math and code blocks preserved.
- `metadata`:
    - `summary`: High-level gist for the Navigator agent.
    - `key_terms`: Array of core concepts for the Evaluator (e.g., `["write-ahead log", "memtable"]`).
    - `complexity`: 1 (Intro) to 10 (Advanced Implementation).
    - `prerequisites`: Array of `node_id`s that must be "Passed" first.
- `exercise_config`:
    - `objective`: The learning goal for this specific node.
    - `skeleton`: The starter code provided to the user.
    - `test_suite`: Hidden unit tests to validate the implementation.

### 2. The Agentic "Socratic" Loop
- **Trigger**: User submits code via `MonacoWrapper.tsx`.
- **Retrieval**: `Researcher` agent doesn't just look at the current node; it looks at `prerequisites` and `parent` summaries.
- **Evaluation**: `Evaluator` runs code in `code_executor.py` (Docker/Sandboxed).
- **Feedback**: If tests fail, `Evaluator` fetches a "Socratic Hint."
    - *Rule*: Never give the answer.
    - *Action*: Refer to a specific line in `node.content` or a sibling node's `summary`.

### 3. Frontend UX Principles (The "IDE" Feel)
- **Resizable Panels**: Use `react-resizable-panels`. Left: Content/Tree; Right: Code/Terminal.
- **Breadcrumb Navigation**: Always visible. Clicking a breadcrumb re-centers the `Navigator` agent.
- **Progressive Loading**: Tree should lazy-load children to handle massive 800+ page books.
- **ThoughtStream**: A small UI component showing what the Agent is currently "thinking" (e.g., "Analyzing your use of the B-Tree leaf split logic...").

### 4. BYOK (Bring Your Own Key) & Privacy
- **Stateless Backend**: The FastAPI backend should not store API keys. 
- **Encryption**: Frontend stores keys in `localStorage` encrypted with a user-defined session password.
- **Provider Support**: OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), and Local (Ollama/vLLM) via OpenAI-compatible endpoints.

## 🛠️ Commands & Workflow
- **Inference Setup**: `export OPENAI_API_BASE="http://localhost:11434/v1"` (for Ollama).
- **Ingestion**: `python 01-backend/core/tree_engine.py --source data/vault/books/raw/ddia.pdf --verbose`
- **Backend**: `cd 01-backend && uvicorn api.main:app --reload`
- **Frontend**: `cd 02-frontend && npm install && npm run dev`

## 📝 Code Style & Interaction
- **Python**: Use `Pydantic` for all data validation. Use `Loguru` for verbose agent tracing.
- **TypeScript**: Strict types only. No `any`. Use `Zod` for API response validation.
- **CSS**: Tailwind for layouts; avoid custom CSS unless for Monaco themes.
- **Git**: Commit messages should reflect the "First Principles" nature (e.g., `feat: implement recursive tree-pruning for navigator agent`).
