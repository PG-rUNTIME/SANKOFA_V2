"""
Document review API: upload and analyse legal documents.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.deps import CurrentUser
from app.services.document_review import review_document
from app.services.schemas import DocumentReviewResult

router = APIRouter(prefix="/documents", tags=["Document Review"])
logger = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
}


@router.post("/review", response_model=DocumentReviewResult)
async def review_uploaded_document(
    file: UploadFile,
    current_user: CurrentUser,
) -> DocumentReviewResult:
    """
    Upload a legal document (PDF or Word) and receive a structured review:
    - Summary
    - Cited cases extracted from the text
    - Citations not found in the legal database
    - Areas for improvement
    - Overall assessment
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_DOCUMENT_EXTENSIONS}",
        )

    # Read and size-check
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    logger.info(
        "Document review requested by %s: %s (%d bytes)",
        current_user.email,
        file.filename,
        len(content),
    )

    try:
        result = await review_document(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("Document review failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document review could not be completed. Please try again.",
        )

    return result
