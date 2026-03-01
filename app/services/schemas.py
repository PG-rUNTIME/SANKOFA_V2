"""
Shared Pydantic schemas used across API and services.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth Schemas ──────────────────────────────────────────────────────────────

class UserCreateRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="user", pattern="^(admin|user)$")

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ── Chat Schemas ──────────────────────────────────────────────────────────────

class Citation(BaseModel):
    url: str
    document_title: str
    source_file: str


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    citations: Optional[List[Citation]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    session_type: str
    is_active: bool
    created_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class NewSessionRequest(BaseModel):
    session_type: str = Field(default="legal_research", pattern="^(legal_research|document_review)$")
    title: Optional[str] = Field(default=None, max_length=512)


class ChatQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[uuid.UUID] = None


# ── Document Review Schemas ───────────────────────────────────────────────────

class DocumentReviewResult(BaseModel):
    summary: str
    cited_cases: List[str]
    non_existing_citations: List[str]
    areas_for_improvement: List[str]
    overall_assessment: str
    disclaimer: str = (
        "This automated review is for research assistance only and does not "
        "constitute legal advice. Always verify with a qualified legal practitioner."
    )
