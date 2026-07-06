"""
Common data models used throughout the plagiarism detection system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np



@dataclass
class Patch:
    """
    Represents one sliding-window patch from a page.
    """

    id: int

    page: int

    x: int
    y: int

    width: int
    height: int

    image: np.ndarray



@dataclass
class Page:
    """
    Represents one page of a document.
    """

    number: int

    image: np.ndarray

    patches: List[Patch] = field(default_factory=list)

    embeddings: Optional[np.ndarray] = None



@dataclass
class Document:
    """
    Represents one uploaded assignment.
    """

    name: str

    pages: List[Page] = field(default_factory=list)

    total_patches: int = 0



@dataclass
class PatchMatch:
    """
    Represents one matched patch.
    """

    source_patch: Patch

    target_patch: Patch

    similarity: float



@dataclass
class PageResult:
    """
    Result of comparing one page.
    """

    page_number: int

    similarity: float

    matches: List[PatchMatch] = field(default_factory=list)

    heatmap: Optional[np.ndarray] = None



@dataclass
class DocumentResult:
    """
    Result of comparing two documents.
    """

    source_document: str

    target_document: str

    similarity: float

    pages: List[PageResult] = field(default_factory=list)




@dataclass
class DetectionResult:
    """
    Final output returned to the dashboard.
    """

    comparisons: List[DocumentResult] = field(default_factory=list)