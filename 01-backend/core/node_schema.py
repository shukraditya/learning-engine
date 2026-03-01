"""Pydantic models for Knowledge Tree nodes.

Defines the schema for all nodes in the hierarchical knowledge tree,
following the ATT "Unit of Truth" specification.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import IntEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class NodeLevel(IntEnum):
    """Hierarchy depth levels for knowledge nodes."""

    ROOT = 0
    CHAPTER = 1
    SECTION = 2
    CONCEPT = 3


class ExerciseConfig(BaseModel):
    """Coding exercise configuration for leaf nodes.

    Contains starter code, hidden tests, and Socratic hints
    that guide without giving answers.
    """

    objective: str = Field(
        ...,
        description="The learning goal for this specific node",
        min_length=10,
        max_length=500,
    )
    skeleton: str = Field(
        ...,
        description="Starter code provided to the user",
    )
    test_suite: str = Field(
        ...,
        description="Hidden unit tests to validate implementation",
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Socratic hints that guide without giving answers",
        max_length=10,
    )
    difficulty: int = Field(
        ...,
        description="Complexity from 1 (intro) to 10 (advanced)",
        ge=1,
        le=10,
    )
    estimated_time_minutes: int = Field(
        ...,
        description="Estimated time to complete the exercise",
        ge=5,
        le=180,
    )

    @field_validator("hints")
    @classmethod
    def validate_hints_socratic(cls, hints: list[str]) -> list[str]:
        """Ensure hints don't contain complete solutions.

        Rules:
        - No code blocks with complete implementations
        - No direct answers like "The answer is X"
        - Questions and guidance only
        """
        forbidden_patterns = [
            r"```\w*\s*\n[^`]*def \w+\([^)]*\):\s*\n\s+[^#].+",  # Complete function
            r"The answer is",
            r"You should use",
            r"Simply",
            r"Just",
        ]

        for hint in hints:
            hint_lower = hint.lower()
            for pattern in forbidden_patterns:
                if re.search(pattern, hint, re.IGNORECASE | re.MULTILINE):
                    raise ValueError(
                        f"Hint appears to contain a solution: '{hint[:50]}...' "
                        f"Hints must be Socratic (guiding questions only)"
                    )
        return hints


class NodeMetadata(BaseModel):
    """Metadata for knowledge nodes.

    Used by Navigator and Evaluator agents for context.
    """

    summary: str = Field(
        ...,
        description="High-level gist for the Navigator agent",
        min_length=20,
        max_length=1000,
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Core concepts for the Evaluator",
        max_length=50,
    )
    complexity: int = Field(
        ...,
        description="Complexity from 1 (intro) to 10 (advanced)",
        ge=1,
        le=10,
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Node IDs that must be 'Passed' first",
    )


class KnowledgeNode(BaseModel):
    """A single node in the hierarchical knowledge tree.

    This is the "Unit of Truth" - every concept, section, and chapter
    is represented as a KnowledgeNode with consistent schema.
    """

    id: str = Field(
        ...,
        description="Unique slug (e.g., 'storage-engine-lsm-compaction')",
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    )
    level: NodeLevel = Field(
        ...,
        description="Depth in hierarchy: ROOT=0, CHAPTER=1, SECTION=2, CONCEPT=3",
    )
    breadcrumb: str = Field(
        ...,
        description="Full path: 'Storage > Hash Indexes > Bitcask'",
    )
    content: str = Field(
        ...,
        description="Cleaned Markdown with LaTeX math and code blocks",
    )
    metadata: NodeMetadata = Field(
        ...,
        description="Node metadata for agent context",
    )
    exercise_config: ExerciseConfig | None = Field(
        default=None,
        description="Exercise for leaf nodes (CONCEPT level)",
    )
    children: list[str] = Field(
        default_factory=list,
        description="Child node IDs (for tree traversal)",
    )
    parent_id: str | None = Field(
        default=None,
        description="Parent node ID (null for ROOT)",
    )
    source_page_start: int | None = Field(
        default=None,
        description="Source PDF page where this node begins",
    )
    source_page_end: int | None = Field(
        default=None,
        description="Source PDF page where this node ends",
    )

    @model_validator(mode="after")
    def validate_leaf_node_has_exercise(self) -> Self:
        """Ensure CONCEPT level nodes have exercise_config."""
        if self.level == NodeLevel.CONCEPT and self.exercise_config is None:
            raise ValueError(
                f"CONCEPT level node '{self.id}' must have exercise_config"
            )
        return self

    @model_validator(mode="after")
    def validate_non_leaf_no_exercise(self) -> Self:
        """Ensure non-leaf nodes don't have exercise_config."""
        if self.level != NodeLevel.CONCEPT and self.exercise_config is not None:
            raise ValueError(
                f"Non-CONCEPT node '{self.id}' should not have exercise_config"
            )
        return self

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows kebab-case convention."""
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(
                f"ID '{v}' must be kebab-case (lowercase, hyphen-separated)"
            )
        return v


class ChapterManifest(BaseModel):
    """Manifest for a processed chapter.

    Contains chapter-level metadata and list of node files.
    """

    book_title: str = Field(..., description="Title of the source book")
    book_author: str = Field(..., description="Author of the source book")
    chapter_number: int = Field(..., ge=1, description="Chapter sequence number")
    chapter_id: str = Field(..., description="Chapter slug identifier")
    chapter_title: str = Field(..., description="Human-readable chapter title")
    total_nodes: int = Field(..., ge=0, description="Total nodes in this chapter")
    leaf_nodes: int = Field(..., ge=0, description="Number of CONCEPT level nodes")
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of processing",
    )
    source_pages: tuple[int, int] = Field(
        ...,
        description="(start_page, end_page) in source PDF",
    )
    nodes: list[KnowledgeNode] = Field(
        ...,
        description="All nodes in this chapter tree",
    )

    def get_root_node(self) -> KnowledgeNode | None:
        """Get the chapter root node (level=CHAPTER)."""
        for node in self.nodes:
            if node.level == NodeLevel.CHAPTER:
                return node
        return None

    def get_leaf_nodes(self) -> list[KnowledgeNode]:
        """Get all CONCEPT level nodes with exercises."""
        return [n for n in self.nodes if n.level == NodeLevel.CONCEPT]


class BookStructure(BaseModel):
    """Output of Pass 1: Book structure extraction.

    Contains TOC and chapter boundaries.
    """

    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    total_pages: int = Field(..., ge=1, description="Total pages in PDF")
    chapters: list[ChapterInfo] = Field(..., description="Chapter metadata")


class ChapterInfo(BaseModel):
    """Metadata for a single chapter."""

    id: str = Field(..., description="Chapter slug (e.g., 'ch01-storage-engines')")
    title: str = Field(..., description="Chapter title")
    start_page: int = Field(..., ge=1, description="First page of chapter")
    end_page: int = Field(..., ge=1, description="Last page of chapter")

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        """Ensure start_page <= end_page."""
        if self.start_page > self.end_page:
            raise ValueError(
                f"Chapter '{self.id}': start_page ({self.start_page}) "
                f"must be <= end_page ({self.end_page})"
            )
        return self


class ChapterStructure(BaseModel):
    """Output of Pass 2: Chapter decomposition.

    Hierarchical tree of sections and concepts.
    """

    chapter_id: str = Field(..., description="Chapter identifier")
    chapter_title: str = Field(..., description="Chapter title")
    sections: list[SectionNode] = Field(..., description="Top-level sections")


class SectionNode(BaseModel):
    """A section within a chapter (Level 2)."""

    id: str = Field(..., description="Section slug")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content (markdown)")
    key_terms: list[str] = Field(default_factory=list)
    subsections: list[ConceptNode] = Field(default_factory=list)
    source_page_start: int | None = None
    source_page_end: int | None = None


class ConceptNode(BaseModel):
    """A concept/subsection (Level 3) - leaf node with exercise."""

    id: str = Field(..., description="Concept slug")
    title: str = Field(..., description="Concept title")
    content: str = Field(..., description="Concept content (markdown)")
    key_terms: list[str] = Field(default_factory=list)
    complexity: int = Field(..., ge=1, le=10)
    prerequisites: list[str] = Field(default_factory=list)
    source_page_start: int | None = None
    source_page_end: int | None = None
