"""
Document Processor - File Upload, OCR, and Analysis Pipeline

Handles PDF, images, and document parsing with OCR support.
Integrates with LLM for content analysis and data extraction.
"""

import os
import io
import json
import hashlib
import tempfile
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class DocumentType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractedData:
    """Extracted data from document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[List[List[str]]] = field(default_factory=list)
    images: List[bytes] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    financial_data: Optional[Dict[str, Any]] = None


@dataclass
class DocumentRecord:
    """Document record with metadata."""
    id: str
    filename: str
    doc_type: DocumentType
    size_bytes: int
    hash: str
    upload_time: datetime
    status: ProcessingStatus = ProcessingStatus.PENDING
    extracted: Optional[ExtractedData] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class OCREngine:
    """OCR engine for text extraction from images."""

    def __init__(self, language: str = "eng"):
        self.language = language
        self._check_availability()

    def _check_availability(self):
        if not HAS_PIL:
            logger.warning("PIL not available - image processing disabled")
        if not HAS_TESSERACT:
            logger.warning("Tesseract not available - OCR disabled")

    def extract_text(self, image_data: bytes) -> str:
        """Extract text from image using OCR."""
        if not HAS_PIL or not HAS_TESSERACT:
            return "[OCR not available]"

        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang=self.language)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_data(self, image_data: bytes) -> Dict[str, Any]:
        """Extract structured data from image."""
        if not HAS_PIL or not HAS_TESSERACT:
            return {"text": "[OCR not available]", "confidence": 0}

        try:
            image = Image.open(io.BytesIO(image_data))
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(c) for c in data["conf"] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            text = pytesseract.image_to_string(image, lang=self.language)

            return {
                "text": text.strip(),
                "confidence": avg_confidence,
                "word_count": len([w for w in data["text"] if w.strip()]),
            }
        except Exception as e:
            logger.error(f"OCR data extraction failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}


class PDFProcessor:
    """PDF document processor."""

    def __init__(self, ocr_engine: Optional[OCREngine] = None):
        self.ocr = ocr_engine or OCREngine()

    def extract(self, pdf_data: bytes) -> ExtractedData:
        """Extract content from PDF."""
        if not HAS_PYMUPDF:
            return ExtractedData(text="[PyMuPDF not available]")

        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text_parts = []
            images = []
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "created": doc.metadata.get("creationDate", ""),
            }

            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

                # Extract images for OCR if text is sparse
                if len(text.strip()) < 100:
                    for img in page.get_images():
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            images.append(image_bytes)

                            # OCR the image
                            ocr_text = self.ocr.extract_text(image_bytes)
                            if ocr_text:
                                text_parts.append(f"[Page {page_num + 1} - OCR]\n{ocr_text}")
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")

            doc.close()

            return ExtractedData(
                text="\n\n".join(text_parts),
                metadata=metadata,
                images=images,
            )

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ExtractedData(text="", metadata={"error": str(e)})


class DocumentProcessor:
    """
    Main document processing pipeline.

    Handles:
    - File upload and storage
    - Type detection
    - Text extraction (PDF, images, text files)
    - OCR for scanned documents
    - Integration with LLM for analysis
    """

    def __init__(
        self,
        upload_dir: str = "/tmp/documents",
        llm_bridge: Optional[Any] = None,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm_bridge
        self.ocr = OCREngine()
        self.pdf_processor = PDFProcessor(self.ocr)
        self.documents: Dict[str, DocumentRecord] = {}

    def _detect_type(self, filename: str, content: bytes) -> DocumentType:
        """Detect document type."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return DocumentType.PDF
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
            return DocumentType.IMAGE
        elif ext in [".txt", ".md", ".rst"]:
            return DocumentType.TEXT
        elif ext == ".json":
            return DocumentType.JSON
        elif ext == ".csv":
            return DocumentType.CSV
        else:
            # Check magic bytes
            if content[:4] == b"%PDF":
                return DocumentType.PDF
            elif content[:3] == b"\xff\xd8\xff":  # JPEG
                return DocumentType.IMAGE
            elif content[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
                return DocumentType.IMAGE
            return DocumentType.UNKNOWN

    def _compute_hash(self, content: bytes) -> str:
        """Compute content hash."""
        return hashlib.sha256(content).hexdigest()

    def upload(
        self,
        filename: str,
        content: bytes,
        tags: Optional[List[str]] = None,
    ) -> DocumentRecord:
        """Upload and process a document."""

        doc_hash = self._compute_hash(content)
        doc_id = f"doc_{doc_hash[:12]}_{int(datetime.now().timestamp())}"
        doc_type = self._detect_type(filename, content)

        # Save file
        file_path = self.upload_dir / doc_id / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

        record = DocumentRecord(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            size_bytes=len(content),
            hash=doc_hash,
            upload_time=datetime.now(),
            status=ProcessingStatus.PENDING,
            tags=tags or [],
        )

        self.documents[doc_id] = record
        return record

    def process(self, doc_id: str) -> DocumentRecord:
        """Process uploaded document."""
        record = self.documents.get(doc_id)
        if not record:
            raise ValueError(f"Document {doc_id} not found")

        record.status = ProcessingStatus.PROCESSING

        try:
            file_path = self.upload_dir / doc_id / record.filename
            content = file_path.read_bytes()

            # Extract based on type
            if record.doc_type == DocumentType.PDF:
                extracted = self.pdf_processor.extract(content)
            elif record.doc_type == DocumentType.IMAGE:
                ocr_result = self.ocr.extract_data(content)
                extracted = ExtractedData(
                    text=ocr_result.get("text", ""),
                    metadata={"ocr_confidence": ocr_result.get("confidence", 0)},
                )
            elif record.doc_type == DocumentType.TEXT:
                extracted = ExtractedData(text=content.decode("utf-8", errors="ignore"))
            elif record.doc_type == DocumentType.JSON:
                data = json.loads(content)
                extracted = ExtractedData(
                    text=json.dumps(data, indent=2),
                    metadata={"json_keys": list(data.keys()) if isinstance(data, dict) else []},
                )
            else:
                extracted = ExtractedData(text=content.decode("utf-8", errors="ignore"))

            record.extracted = extracted
            record.status = ProcessingStatus.COMPLETED

        except Exception as e:
            record.status = ProcessingStatus.FAILED
            record.error = str(e)
            logger.error(f"Processing failed for {doc_id}: {e}")

        return record

    def analyze_with_llm(
        self,
        doc_id: str,
        analysis_type: str = "general",
    ) -> DocumentRecord:
        """Analyze document with LLM."""
        record = self.documents.get(doc_id)
        if not record:
            raise ValueError(f"Document {doc_id} not found")

        if not record.extracted:
            self.process(doc_id)

        if not self.llm:
            record.llm_analysis = {"error": "LLM not configured"}
            return record

        try:
            if analysis_type == "financial":
                response = self.llm.analyze_financial(record.extracted.text)
            else:
                response = self.llm.analyze_document(
                    record.extracted.text,
                    doc_type=analysis_type,
                )

            # Parse JSON response if possible
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                analysis = {"summary": response.content}

            record.llm_analysis = {
                "type": analysis_type,
                "result": analysis,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            record.llm_analysis = {"error": str(e)}
            logger.error(f"LLM analysis failed for {doc_id}: {e}")

        return record

    def batch_analyze(
        self,
        doc_ids: List[str],
        analysis_type: str = "financial",
    ) -> Dict[str, Any]:
        """Analyze multiple documents and create synthesis."""
        results = []

        for doc_id in doc_ids:
            record = self.analyze_with_llm(doc_id, analysis_type)
            if record.llm_analysis and "error" not in record.llm_analysis:
                results.append({
                    "doc_id": doc_id,
                    "filename": record.filename,
                    "analysis": record.llm_analysis.get("result"),
                })

        # Create synthesis if LLM available
        synthesis = None
        if self.llm and results:
            try:
                synthesis_prompt = f"Synthesize these {len(results)} document analyses:\n"
                synthesis_prompt += json.dumps(results, indent=2)

                response = self.llm.generate(
                    prompt=synthesis_prompt,
                    system="Create a comprehensive synthesis of multiple document analyses. Identify patterns, totals, and key insights.",
                )
                synthesis = response.content
            except Exception as e:
                synthesis = f"Synthesis failed: {e}"

        return {
            "documents": results,
            "synthesis": synthesis,
            "total_documents": len(doc_ids),
            "successful": len(results),
        }

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get document record."""
        return self.documents.get(doc_id)

    def list_documents(
        self,
        status: Optional[ProcessingStatus] = None,
        doc_type: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[DocumentRecord]:
        """List documents with optional filters."""
        results = list(self.documents.values())

        if status:
            results = [d for d in results if d.status == status]
        if doc_type:
            results = [d for d in results if d.doc_type == doc_type]
        if tags:
            results = [d for d in results if any(t in d.tags for t in tags)]

        return sorted(results, key=lambda d: d.upload_time, reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and files."""
        if doc_id not in self.documents:
            return False

        # Remove files
        doc_dir = self.upload_dir / doc_id
        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)

        del self.documents[doc_id]
        return True
