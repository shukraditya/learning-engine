"""Tree Decomposer - Multi-pass LLM pipeline for textbook processing.

Orchestrates the transformation of PDF textbooks into hierarchical
Knowledge Trees using a 3-pass architecture:
  Pass 1: Book structure extraction (TOC/chapters)
  Pass 2: Chapter decomposition (sections/concepts)
  Pass 3: Exercise generation (coding exercises for leaf nodes)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.node_schema import (
    BookStructure,
    ChapterInfo,
    ChapterManifest,
    ChapterStructure,
    ConceptNode,
    ExerciseConfig,
    KnowledgeNode,
    NodeLevel,
    NodeMetadata,
    SectionNode,
)
from services.llm_factory import LLMClient, LLMConfig
from services.local_llm import LocalLLMWrapper


@dataclass
class ProcessingContext:
    """Context passed through the pipeline."""

    book_slug: str
    output_dir: Path
    llm_client: LLMClient
    cache_dir: Path | None = None
    verbose: bool = False


class DocumentProcessor:
    """Extract structured content from PDF using Marker, exporting to Markdown."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def pdf_to_markdown(self, path: Path, output_path: Path | None = None) -> tuple[dict, str]:
        """Convert PDF to Markdown using Marker.

        Args:
            path: Path to PDF file
            output_path: Optional path to save Markdown file

        Returns:
            Tuple of (page metadata dict, Markdown string)
        """
        logger.info(f"Converting PDF to Markdown: {path}")

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            # Create converter with default models
            converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )

            # Convert PDF
            rendered = converter(str(path))

            # Extract markdown text
            md_content, _, _ = text_from_rendered(rendered)

            # Get page count from metadata if available
            page_count = rendered.metadata.get("page_count", len(md_content) // 3000 + 1)
            pages_meta = {"page_count": page_count}

            if self.verbose:
                logger.debug(
                    f"Document: {page_count} pages, "
                    f"md: {len(md_content)} chars"
                )

            # Save to file if output_path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                logger.info(f"Saved Markdown: {output_path}")

            return pages_meta, md_content

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

    def extract_text_by_pages(self, pages_meta: dict) -> dict[int, str]:
        """Extract text organized by page number.

        Args:
            pages_meta: Page metadata from Marker

        Returns:
            Dict mapping page_number -> page_text
        """
        pages: dict[int, str] = {}
        page_list = pages_meta.get("pages", [])

        for i, page_text in enumerate(page_list, start=1):
            pages[i] = page_text if isinstance(page_text, str) else str(page_text)

        return pages

    def get_first_n_pages(self, pages_meta: dict, n: int) -> str:
        """Get concatenated text from first N pages (for TOC extraction)."""
        pages = self.extract_text_by_pages(pages_meta)
        sorted_pages = sorted(pages.items())
        return "\n\n".join(text for _, text in sorted_pages[:n])


class BookStructureExtractionAgent:
    """Pass 1: Extract TOC and chapter boundaries.

    Input: Raw document text (first ~50 pages of Markdown)
    Output: BookStructure with title, author, chapters[]
    """

    SYSTEM_PROMPT = """You are analyzing a CS textbook. Extract the table of contents structure.

Rules:
- Identify all chapters with their exact titles
- Estimate page boundaries for each chapter
- Create URL-friendly slugs for chapter IDs (kebab-case)
- Be precise with page numbers - look for page markers in the text"""

    def __init__(self, llm_client: LLMClient, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose = verbose

    async def extract_from_markdown(
        self, pages_meta: dict, full_md: str, first_n_pages: int = 50
    ) -> BookStructure:
        """Extract book structure from Markdown content.

        Args:
            pages_meta: Page metadata dict with page_count
            full_md: Full book Markdown content
            first_n_pages: Number of pages to scan for TOC

        Returns:
            BookStructure with chapters list
        """
        logger.info(f"Pass 1: Extracting book structure (first {first_n_pages} pages)")

        # Extract first N pages from Markdown
        lines = full_md.split("\n")
        lines_per_page = len(lines) / pages_meta.get("page_count", 1)
        end_line = int(first_n_pages * lines_per_page)
        toc_text = "\n".join(lines[:end_line])

        # Get total pages
        total_pages = pages_meta.get("page_count", 0)

        user_prompt = f"""Extract the book structure from this table of contents:

---TOC TEXT---
{toc_text[:15000]}
---END TOC---

Total pages in document: {total_pages}

Output JSON matching this schema:
{{
  "title": "Book Title",
  "author": "Author Name",
  "total_pages": {total_pages},
  "chapters": [
    {{
      "id": "ch01-chapter-slug",
      "title": "Full Chapter Title",
      "start_page": 1,
      "end_page": 25
    }}
  ]
}}

Requirements:
- Chapter IDs: kebab-case, include number prefix (ch01-, ch02-, etc.)
- Page ranges: contiguous, no gaps between chapters
- Last chapter ends at page {total_pages}"""

        response = await self.llm_client.complete(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            structure = BookStructure.model_validate(data)

            logger.info(
                f"Extracted: '{structure.title}' by {structure.author}, "
                f"{len(structure.chapters)} chapters"
            )
            return structure

        except Exception as e:
            logger.error(f"Failed to parse book structure: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            raise


class ChapterDecompositionAgent:
    """Pass 2: Decompose chapter into hierarchical sections/concepts.

    Input: Chapter Markdown content + book context
    Output: ChapterStructure with sections/subsections tree
    """

    SYSTEM_PROMPT = """You are decomposing a textbook chapter (in Markdown format) into hierarchical knowledge nodes.

Hierarchy Rules:
- Level 1 (CHAPTER): Chapter title (already provided)
- Level 2 (SECTION): Major sections (## headers in the Markdown)
- Level 3 (CONCEPT): Subsections (### headers) or specific concepts

For each node:
- Extract clean markdown content preserving code blocks and math
- Identify 3-8 key_terms per node
- Identify prerequisites from earlier in the chapter

Content Guidelines:
- Preserve LaTeX math expressions ($...$ or $$...$$)
- Keep code blocks with language tags
- Maintain heading structure (##, ###)
- Remove page headers/footers/noise if present
- Each section/concept should have substantial content (not just headers)"""

    def __init__(self, llm_client: LLMClient, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose = verbose

    async def decompose(
        self,
        chapter_md: str,
        chapter_info: ChapterInfo,
        book_context: dict[str, Any],
    ) -> ChapterStructure:
        """Decompose chapter Markdown into hierarchical structure.

        Args:
            chapter_md: Chapter content in Markdown format
            chapter_info: Chapter metadata from Pass 1
            book_context: Book title, author, etc.

        Returns:
            ChapterStructure with section/concept tree
        """
        logger.info(f"Pass 2: Decomposing {chapter_info.id}")

        # Truncate if extremely long
        max_chars = 40000
        if len(chapter_md) > max_chars:
            logger.warning(
                f"Chapter {chapter_info.id} is very long ({len(chapter_md)} chars), "
                f"truncating to {max_chars}"
            )
            chapter_md = chapter_md[:max_chars] + "\n\n[Content truncated...]"

        user_prompt = f"""Decompose this Markdown chapter into hierarchical knowledge nodes.

Book: {book_context.get('title', 'Unknown')}
Chapter: {chapter_info.title} (Chapter {chapter_info.id})

---CHAPTER MARKDOWN---
{chapter_md}
---END MARKDOWN---

Output JSON matching this schema:
{{
  "chapter_id": "{chapter_info.id}",
  "chapter_title": "{chapter_info.title}",
  "sections": [
    {{
      "id": "section-slug",
      "title": "Section Title",
      "content": "Full markdown content including ## header and all subsections until next ##...",
      "key_terms": ["term1", "term2"],
      "subsections": [
        {{
          "id": "concept-slug",
          "title": "Concept Title",
          "content": "Concept markdown content (### header and content)...",
          "key_terms": ["concept_term"],
          "complexity": 5,
          "prerequisites": ["earlier-concept-id"]
        }}
      ]
    }}
  ]
}}

Important:
- Section IDs: descriptive slugs (e.g., "btree-fundamentals")
- Concept IDs: just the slug (e.g., "node-structure") - the chapter prefix will be added automatically
- Prerequisites: reference other concept IDs (without chapter prefix if in same chapter)
- Content should include the Markdown headers and all relevant content
- Preserve code blocks and math expressions exactly as they appear"""

        response = await self.llm_client.complete(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            structure = ChapterStructure.model_validate(data)

            section_count = len(structure.sections)
            concept_count = sum(len(s.subsections) for s in structure.sections)
            logger.info(
                f"Decomposed {chapter_info.id}: {section_count} sections, "
                f"{concept_count} concepts"
            )
            return structure

        except Exception as e:
            logger.error(f"Failed to parse chapter structure: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            raise


class ExerciseGenerationAgent:
    """Pass 3: Generate coding exercises for leaf nodes.

    Input: Concept node content + prerequisites context
    Output: ExerciseConfig with objective, skeleton, tests, hints
    """

    SYSTEM_PROMPT = """You are generating coding exercises for CS concepts from textbooks.

Your exercises must:
1. Test understanding of the specific concept (not generic coding)
2. Include starter code with clear TODOs
3. Have comprehensive hidden tests
4. Provide Socratic hints (guiding questions, NOT answers)

Socratic Hint Rules:
- Ask questions that lead to understanding
- Reference specific properties or constraints from the concept
- NEVER provide complete implementations
- NEVER give away the algorithm or data structure directly
- Use phrases like "Consider...", "What would happen if...", "Recall that..."

Example good hint: "Consider what happens to the tree height when a node splits. How does the B-Tree maintain balance?"
Example bad hint: "Use the split_node() function and divide keys evenly."

Test Design:
- Tests should verify correctness, not implementation details
- Include edge cases (empty input, single element, maximum capacity)
- Tests should be runnable with pytest"""

    PYTHON_SKELETON_TEMPLATE = '''# TODO: Implement according to the objective
# Objective: {objective}

def {function_name}({params}):
    """
    {docstring}
    """
    # Your implementation here
    pass
'''

    def __init__(self, llm_client: LLMClient, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose = verbose

    async def generate(
        self,
        concept: ConceptNode,
        chapter_context: str,
        prerequisite_nodes: list[KnowledgeNode],
    ) -> ExerciseConfig:
        """Generate exercise for a concept node.

        Args:
            concept: The concept node to generate exercise for
            chapter_context: Surrounding chapter content for context
            prerequisite_nodes: Previously learned prerequisite nodes

        Returns:
            ExerciseConfig with objective, skeleton, tests, hints
        """
        logger.debug(f"Pass 3: Generating exercise for {concept.id}")

        prereq_summary = "\n".join(
            f"- {n.id}: {n.metadata.summary[:100]}..."
            for n in prerequisite_nodes[:3]  # Limit context
        )

        user_prompt = f"""Generate a coding exercise for this CS concept:

Concept: {concept.title}
Complexity: {concept.complexity}/10
Content:
{concept.content[:5000]}

Prerequisites:
{prereq_summary or "None"}

Output JSON matching this schema:
{{
  "objective": "Clear learning goal",
  "skeleton": "Python starter code with TODOs",
  "test_suite": "pytest test functions",
  "hints": [
    "Socratic hint 1 (guiding question)",
    "Socratic hint 2 (reference concept property)"
  ],
  "difficulty": {concept.complexity},
  "estimated_time_minutes": 30
}}

Requirements:
- objective: One sentence, action-oriented (e.g., "Implement a B-Tree node split")
- skeleton: Python function stub with clear signature and docstring
- test_suite: Complete pytest test file with 3-5 test cases
- hints: 2-4 Socratic hints, NO CODE, only questions/guidance
- difficulty: Match concept complexity ({concept.complexity})
- estimated_time_minutes: 15-90 based on complexity"""

        response = await self.llm_client.complete(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            exercise = ExerciseConfig.model_validate(data)

            logger.debug(
                f"Generated exercise for {concept.id}: "
                f"difficulty={exercise.difficulty}, "
                f"hints={len(exercise.hints)}"
            )
            return exercise

        except Exception as e:
            logger.error(f"Failed to parse exercise for {concept.id}: {e}")
            # Return fallback exercise
            return self._fallback_exercise(concept)

    def _fallback_exercise(self, concept: ConceptNode) -> ExerciseConfig:
        """Create a minimal fallback exercise if LLM fails."""
        return ExerciseConfig(
            objective=f"Review and implement concepts from {concept.title}",
            skeleton=f'# TODO: Review {concept.title}\n\ndef review_concept():\n    """\n    Based on the text, implement the key ideas from {concept.title}.\n    """\n    pass\n',
            test_suite=f'import pytest\n\ndef test_{concept.id.replace("-", "_")}():\n    # TODO: Add tests based on concept content\n    assert True\n',
            hints=[
                f"Review the key properties of {concept.title} described in the text.",
                "What are the main constraints or invariants?",
            ],
            difficulty=concept.complexity,
            estimated_time_minutes=30,
        )


class TreeBuilder:
    """Assemble hierarchical nodes with parent-child links."""

    def __init__(self, book_slug: str, verbose: bool = False):
        self.book_slug = book_slug
        self.verbose = verbose
        self.nodes: dict[str, KnowledgeNode] = {}

    def build_chapter_tree(
        self,
        chapter_structure: ChapterStructure,
        chapter_info: ChapterInfo,
    ) -> list[KnowledgeNode]:
        """Build KnowledgeNode tree from ChapterStructure.

        Args:
            chapter_structure: Output from Pass 2
            chapter_info: Chapter metadata

        Returns:
            List of all nodes in the chapter tree
        """
        self.nodes = {}

        # Create chapter root node (Level 1)
        chapter_node = KnowledgeNode(
            id=chapter_info.id,
            level=NodeLevel.CHAPTER,
            breadcrumb=chapter_structure.chapter_title,
            content=f"# {chapter_structure.chapter_title}\n\nChapter overview.",
            metadata=NodeMetadata(
                summary=f"Chapter on {chapter_structure.chapter_title}",
                key_terms=[],
                complexity=5,
                prerequisites=[],
            ),
            exercise_config=None,
            children=[],
            parent_id=None,
            source_page_start=chapter_info.start_page,
            source_page_end=chapter_info.end_page,
        )
        self.nodes[chapter_node.id] = chapter_node

        # Build sections and concepts
        for section in chapter_structure.sections:
            self._build_section(section, chapter_node.id, chapter_structure.chapter_title)

        # Update chapter children
        chapter_node.children = [
            n.id for n in self.nodes.values() if n.parent_id == chapter_node.id
        ]

        nodes_list = list(self.nodes.values())
        logger.debug(f"Built tree: {len(nodes_list)} nodes for {chapter_info.id}")
        return nodes_list

    def _build_section(
        self, section: SectionNode, parent_id: str, breadcrumb_prefix: str
    ) -> None:
        """Build section node and its concept children."""
        section_id = f"{parent_id}-{section.id}"
        breadcrumb = f"{breadcrumb_prefix} > {section.title}"

        section_node = KnowledgeNode(
            id=section_id,
            level=NodeLevel.SECTION,
            breadcrumb=breadcrumb,
            content=section.content,
            metadata=NodeMetadata(
                summary=f"Section on {section.title}",
                key_terms=section.key_terms,
                complexity=5,
                prerequisites=[],
            ),
            exercise_config=None,
            children=[],
            parent_id=parent_id,
            source_page_start=section.source_page_start,
            source_page_end=section.source_page_end,
        )
        self.nodes[section_node.id] = section_node

        # Build concept children
        for concept in section.subsections:
            self._build_concept(concept, section_node.id, breadcrumb)

        # Update section children
        section_node.children = [
            n.id for n in self.nodes.values() if n.parent_id == section_node.id
        ]

    def _build_concept(
        self, concept: ConceptNode, parent_id: str, breadcrumb_prefix: str
    ) -> None:
        """Build concept leaf node with exercise placeholder."""
        concept_id = f"{parent_id}-{concept.id}"
        breadcrumb = f"{breadcrumb_prefix} > {concept.title}"

        concept_node = KnowledgeNode(
            id=concept_id,
            level=NodeLevel.CONCEPT,
            breadcrumb=breadcrumb,
            content=concept.content,
            metadata=NodeMetadata(
                summary=f"Concept: {concept.title}",
                key_terms=concept.key_terms,
                complexity=concept.complexity,
                prerequisites=concept.prerequisites,
            ),
            exercise_config=None,  # Will be filled by Pass 3
            children=[],  # Leaf node
            parent_id=parent_id,
            source_page_start=concept.source_page_start,
            source_page_end=concept.source_page_end,
        )
        self.nodes[concept_node.id] = concept_node

    def link_prerequisites(
        self, all_nodes: list[KnowledgeNode], current_chapter_num: int
    ) -> list[KnowledgeNode]:
        """Link prerequisites across chapters.

        Args:
            all_nodes: All nodes from all chapters processed so far
            current_chapter_num: Current chapter number

        Returns:
            Updated nodes with prerequisite links
        """
        # Build ID lookup
        node_map = {n.id: n for n in all_nodes}

        # For each concept node, resolve prerequisites
        for node in all_nodes:
            if node.level != NodeLevel.CONCEPT:
                continue

            resolved_prereqs = []
            for prereq_id in node.metadata.prerequisites:
                # If full ID exists, use it
                if prereq_id in node_map:
                    resolved_prereqs.append(prereq_id)
                else:
                    # Try to find in earlier chapters
                    for earlier_node in all_nodes:
                        if earlier_node.id.endswith(prereq_id):
                            resolved_prereqs.append(earlier_node.id)
                            break

            node.metadata.prerequisites = resolved_prereqs

        return all_nodes


class ChapterWriter:
    """Write chapter data to JSON files."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose

    def write_chapter(
        self,
        nodes: list[KnowledgeNode],
        chapter_info: ChapterInfo,
        book_structure: BookStructure,
    ) -> Path:
        """Write chapter nodes to JSON file.

        Args:
            nodes: All nodes in the chapter
            chapter_info: Chapter metadata
            book_structure: Full book structure

        Returns:
            Path to written file
        """
        # Create book directory
        book_slug = self._slugify(book_structure.title)
        book_dir = self.output_dir / book_slug
        book_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        leaf_count = sum(1 for n in nodes if n.level == NodeLevel.CONCEPT)
        manifest = ChapterManifest(
            book_title=book_structure.title,
            book_author=book_structure.author,
            chapter_number=self._extract_chapter_number(chapter_info.id),
            chapter_id=chapter_info.id,
            chapter_title=chapter_info.title,
            total_nodes=len(nodes),
            leaf_nodes=leaf_count,
            source_pages=(chapter_info.start_page, chapter_info.end_page),
            nodes=nodes,
        )

        # Write file
        chapter_num = manifest.chapter_number
        file_path = book_dir / f"chapter-{chapter_num:03d}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Wrote {file_path}: {len(nodes)} nodes")
        return file_path

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")

    def _extract_chapter_number(self, chapter_id: str) -> int:
        """Extract number from chapter ID like 'ch01-xxx'."""
        match = re.search(r"ch(\d+)", chapter_id)
        if match:
            return int(match.group(1))
        return 1


class TreeEngine:
    """Main orchestrator for the multi-pass tree decomposition pipeline."""

    # Class references for external access
    tree_builder_class = TreeBuilder
    writer_class = ChapterWriter

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        output_dir: Path | None = None,
        cache_dir: Path | None = None,
        verbose: bool = False,
        local_llm = None,
    ):
        """Initialize the tree engine.

        Args:
            llm_config: LLM configuration (uses env defaults if None)
            output_dir: Output directory for processed books
            cache_dir: Cache directory for LLM responses
            verbose: Enable debug logging
            local_llm: Optional local GPU LLM instance
        """
        self.verbose = verbose
        self.output_dir = output_dir or Path("03-data/vault/books/processed")
        self.cache_dir = cache_dir
        self._local_llm = local_llm

        # Initialize LLM client
        if local_llm:
            # Use local LLM wrapper
            self.llm_client = LocalLLMWrapper(local_llm)
        elif llm_config:
            self.llm_client = LLMClient(llm_config)
        else:
            from services.llm_factory import LLMFactory

            self.llm_client = LLMFactory.from_env()

        # Initialize agents
        self.doc_processor = DocumentProcessor(verbose=verbose)
        self.structure_agent = BookStructureExtractionAgent(
            self.llm_client, verbose=verbose
        )
        self.decomposition_agent = ChapterDecompositionAgent(
            self.llm_client, verbose=verbose
        )
        self.exercise_agent = ExerciseGenerationAgent(
            self.llm_client, verbose=verbose
        )

        # Track processed nodes for prerequisite linking
        self.all_nodes: list[KnowledgeNode] = []

        if verbose:
            logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    async def process_book(
        self,
        pdf_path: Path,
        max_chapters: int | None = None,
        skip_exercises: bool = False,
    ) -> list[Path]:
        """Process a PDF book through the full pipeline.

        Flow:
        1. PDF -> Markdown (Docling)
        2. Save full book Markdown
        3. Extract book structure (TOC) from first pages
        4. Split Markdown by chapters (LLM-assisted)
        5. Process each chapter Markdown -> Tree

        Args:
            pdf_path: Path to PDF file
            max_chapters: Limit processing to first N chapters (for testing)
            skip_exercises: Skip Pass 3 (exercise generation)

        Returns:
            List of paths to written chapter files
        """
        logger.info(f"Starting tree decomposition: {pdf_path}")

        # Pass 0: PDF -> Markdown
        book_slug = self._slugify(pdf_path.stem)
        md_output_path = self.output_dir / book_slug / f"{book_slug}.md"

        pages_meta, full_md = self.doc_processor.pdf_to_markdown(
            pdf_path, output_path=md_output_path
        )

        # Pass 1: Book structure extraction (uses first N pages for TOC)
        book_structure = await self.structure_agent.extract_from_markdown(
            pages_meta, full_md
        )

        # Save structure for reference
        structure_path = self.output_dir / book_slug / "structure.json"
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(book_structure.model_dump(), f, indent=2, default=str)
        logger.info(f"Saved book structure: {structure_path}")

        # Limit chapters if specified
        chapters = book_structure.chapters
        if max_chapters:
            chapters = chapters[:max_chapters]
            logger.info(f"Limited to first {max_chapters} chapters")

        # Pass 1.5: Split Markdown by chapters
        # This uses page boundaries as hints to split the Markdown
        chapter_mds = self.split_markdown_by_chapters(
            full_md, book_structure, pages_meta
        )

        # Process each chapter
        output_files: list[Path] = []

        for idx, chapter_info in enumerate(chapters):
            chapter_num = idx + 1
            logger.info(f"Processing chapter {chapter_num}/{len(chapters)}: {chapter_info.id}")

            try:
                chapter_md = chapter_mds.get(chapter_info.id, "")
                if not chapter_md:
                    logger.warning(f"No Markdown content for {chapter_info.id}, skipping")
                    continue

                # Save chapter Markdown
                chapter_md_path = (
                    self.output_dir / book_slug / "chapters" / f"{chapter_info.id}.md"
                )
                chapter_md_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chapter_md_path, "w", encoding="utf-8") as f:
                    f.write(chapter_md)

                # Pass 2: Chapter decomposition (now works with Markdown)
                chapter_structure = await self.decomposition_agent.decompose(
                    chapter_md=chapter_md,
                    chapter_info=chapter_info,
                    book_context={
                        "title": book_structure.title,
                        "author": book_structure.author,
                    },
                )

                # Build tree
                tree_builder = TreeBuilder(book_slug, verbose=self.verbose)
                nodes = tree_builder.build_chapter_tree(chapter_structure, chapter_info)

                # Pass 3: Exercise generation (optional)
                if not skip_exercises:
                    nodes = await self._generate_exercises(nodes, chapter_md)

                # Link prerequisites
                self.all_nodes.extend(nodes)
                nodes = tree_builder.link_prerequisites(self.all_nodes, chapter_num)

                # Write chapter JSON
                writer = ChapterWriter(self.output_dir, verbose=self.verbose)
                file_path = writer.write_chapter(nodes, chapter_info, book_structure)
                output_files.append(file_path)

            except Exception as e:
                logger.error(f"Failed to process chapter {chapter_info.id}: {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()
                continue

        logger.info(f"Completed: {len(output_files)} chapters written")
        return output_files

    def split_markdown_by_chapters(
        self, full_md: str, book_structure: BookStructure, document: "Document"
    ) -> dict[str, str]:
        """Split full Markdown into chapter-specific chunks.

        Uses a heuristic approach:
        1. Estimate character positions from page numbers
        2. Look for chapter headings near those positions
        3. Split Markdown at chapter boundaries

        Args:
            full_md: Full book Markdown
            book_structure: Book structure with chapter page boundaries
            pages_meta: Page metadata dict with page_count

        Returns:
            Dict mapping chapter_id -> chapter_markdown
        """
        lines = full_md.split("\n")
        total_lines = len(lines)
        total_pages = pages_meta.get("page_count", 1)

        # Estimate lines per page (rough approximation)
        lines_per_page = total_lines / total_pages if total_pages > 0 else 40

        chapter_mds = {}
        chapter_boundaries = []

        for i, ch in enumerate(book_structure.chapters):
            # Estimate line number from page number
            start_line = int((ch.start_page - 1) * lines_per_page)

            # Look for chapter heading near this position
            search_start = max(0, start_line - 20)
            search_end = min(total_lines, start_line + 50)

            # Find the best matching heading
            chapter_line = start_line
            for line_num in range(search_start, search_end):
                line = lines[line_num] if line_num < total_lines else ""
                # Match common chapter heading patterns
                if line.startswith(("# ", "## ")):
                    ch_title_clean = ch.title.lower().replace(":", "").strip()
                    line_clean = line.lower().replace("#", "").replace(":", "").strip()
                    # Check if heading contains chapter title (fuzzy match)
                    if any(word in line_clean for word in ch_title_clean.split()[:3]):
                        chapter_line = line_num
                        break

            chapter_boundaries.append({
                "id": ch.id,
                "title": ch.title,
                "start_line": chapter_line,
                "end_page": ch.end_page,
            })

        # Now extract content between boundaries
        for i, boundary in enumerate(chapter_boundaries):
            start = boundary["start_line"]

            # End is either the next chapter's start or end of document
            if i + 1 < len(chapter_boundaries):
                end = chapter_boundaries[i + 1]["start_line"]
            else:
                end = total_lines

            chapter_content = "\n".join(lines[start:end])
            chapter_mds[boundary["id"]] = chapter_content

            if self.verbose:
                logger.debug(
                    f"Chapter {boundary['id']}: lines {start}-{end} "
                    f"({len(chapter_content)} chars)"
                )

        return chapter_mds

    async def _generate_exercises(
        self, nodes: list[KnowledgeNode], chapter_context: str
    ) -> list[KnowledgeNode]:
        """Run Pass 3: Generate exercises for all concept nodes."""
        concept_nodes = [n for n in nodes if n.level == NodeLevel.CONCEPT]

        # Get prerequisite nodes (from earlier processing)
        prereq_map = {n.id: n for n in self.all_nodes}

        tasks = []
        for node in concept_nodes:
            # Create ConceptNode from KnowledgeNode
            concept = ConceptNode(
                id=node.id.split("-")[-1],  # Last part of ID
                title=node.breadcrumb.split(">")[-1].strip(),
                content=node.content,
                key_terms=node.metadata.key_terms,
                complexity=node.metadata.complexity,
                prerequisites=node.metadata.prerequisites,
                source_page_start=node.source_page_start,
                source_page_end=node.source_page_end,
            )

            # Get prerequisite nodes
            prereq_nodes = [
                prereq_map[p] for p in node.metadata.prerequisites if p in prereq_map
            ]

            task = self.exercise_agent.generate(concept, chapter_context, prereq_nodes)
            tasks.append((node, task))

        # Run exercises in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent LLM calls

        async def run_with_limit(node: KnowledgeNode, task):
            async with semaphore:
                try:
                    exercise = await task
                    node.exercise_config = exercise
                    return node
                except Exception as e:
                    logger.error(f"Exercise generation failed for {node.id}: {e}")
                    return node

        results = await asyncio.gather(
            *[run_with_limit(node, task) for node, task in tasks]
        )

        # Update nodes with exercises
        exercise_map = {n.id: n.exercise_config for n in results}
        for node in nodes:
            if node.id in exercise_map:
                node.exercise_config = exercise_map[node.id]

        logger.info(f"Generated {len(results)} exercises")
        return nodes

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")
