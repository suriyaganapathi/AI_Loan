"""
AI Calling Views
================
API endpoints for AI-powered calling functionality
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import glob
import json

from app.ai_calling.service import (
    make_outbound_call,
    get_call_data_store,
    gemini_client
)
from config import settings


router = APIRouter()


# ============================================================
# LANGUAGE MAPPING FUNCTION
# ============================================================

def normalize_language(language: str) -> str:
    """
    Convert language names to locale codes
    HINDI/Hindi/hindi -> hi-IN
    ENGLISH/English/english -> en-IN
    TAMIL/Tamil/tamil -> ta-IN
    """
    language_upper = language.upper().strip()
    
    language_map = {
        "HINDI": "hi-IN",
        "ENGLISH": "en-IN",
        "TAMIL": "ta-IN"
    }
    
    # If it's already a locale code, return it
    if language in settings.LANGUAGE_CONFIG:
        return language
    
    # Otherwise, map it
    return language_map.get(language_upper, language)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class BorrowerInfo(BaseModel):
    """Single borrower information"""
    borrower_id: str = Field(..., description="Unique identifier for the borrower")
    phone_number: str = Field(..., description="Phone number with country code (e.g., +911234567890)")
    preferred_language: str = Field(default="en-IN", description="Preferred language: en-IN, hi-IN, or ta-IN")
    
    class Config:
        json_schema_extra = {
            "example": {
                "NO": "BRW123456",
                "cell1": "+911234567890",
                "preferred_language": "hi-IN"
            }
        }


class BulkCallRequest(BaseModel):
    """Request model for bulk calling"""
    borrowers: List[BorrowerInfo] = Field(..., description="List of borrowers to call")
    
    class Config:
        json_schema_extra = {
            "example": {
                "borrowers": [
                    {
                        "NO": "BRW123456",
                        "cell1": "+911234567890",
                        "preferred_language": "hi-IN"
                    },
                    {
                        "NO": "BRW789012",
                        "cell1": "+911987654321",
                        "preferred_language": "en-IN"
                    },
                    {
                        "NO": "BRW345678",
                        "cell1": "+911122334455",
                        "preferred_language": "ta-IN"
                    }
                ]
            }
        }


class SingleCallRequest(BaseModel):
    """Request model for single call"""
    to_number: str = Field(..., description="Phone number to call (with country code)")
    language: str = Field(default="en-IN", description="Preferred language: en-IN, hi-IN, or ta-IN")
    borrower_id: Optional[str] = Field(None, description="Optional borrower ID for tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "to_number": "+911234567890",
                "language": "hi-IN",
                "borrower_id": "BRW123456"
            }
        }


class CallResponse(BaseModel):
    """Response model for call initiation"""
    success: bool
    call_uuid: Optional[str] = None
    status: Optional[str] = None
    to_number: Optional[str] = None
    language: Optional[str] = None
    borrower_id: Optional[str] = None
    error: Optional[str] = None


class BulkCallResponse(BaseModel):
    """Response model for bulk calling"""
    total_requests: int
    successful_calls: int
    failed_calls: int
    results: List[CallResponse]


# ============================================================
# API ENDPOINTS
# ============================================================

@router.get("/")
async def ai_calling_root():
    """AI Calling module root endpoint"""
    return {
        "message": "AI Calling Module",
        "status": "active",
        "supported_languages": list(settings.LANGUAGE_CONFIG.keys()),
        "features": [
            "Multi-language support (English, Hindi, Tamil)",
            "Automatic language detection",
            "AI-powered conversation analysis",
            "Bulk calling capability",
            "Real-time transcription",
            "Sentiment analysis",
            "Intent classification"
        ]
    }


@router.post("/trigger_calls", response_model=BulkCallResponse)
async def trigger_bulk_calls(request: BulkCallRequest):
    """
    Trigger multiple calls to borrowers with their preferred languages
    
    This endpoint allows you to initiate multiple AI-powered calls in one request.
    Each borrower can have a different preferred language (English, Hindi, or Tamil).
    
    **Input:**
    - List of borrowers with their IDs, phone numbers, and preferred languages
    
    **Output:**
    - Summary of successful and failed calls
    - Individual call UUIDs for tracking
    
    **Supported Languages:**
    - en-IN: English (India)
    - hi-IN: Hindi
    - ta-IN: Tamil
    """
    
    if not request.borrowers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No borrowers provided in the request"
        )
    
    results = []
    successful = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"📞 BULK CALL REQUEST - {len(request.borrowers)} borrowers")
    print(f"{'='*60}\n")
    
    for borrower in request.borrowers:
        print(f"Processing borrower: {borrower.borrower_id}")
        print(f"  Phone: {borrower.phone_number}")
        print(f"  Language: {borrower.preferred_language}")
        
        # Normalize language (HINDI -> hi-IN, English -> en-IN, etc.)
        normalized_language = normalize_language(borrower.preferred_language)
        print(f"  Normalized Language: {normalized_language}")
        
        # Validate language
        if normalized_language not in settings.LANGUAGE_CONFIG:
            result = CallResponse(
                success=False,
                error=f"Unsupported language: {borrower.preferred_language}",
                to_number=borrower.phone_number,
                borrower_id=borrower.borrower_id,
                language=borrower.preferred_language
            )
            results.append(result)
            failed += 1
            continue
        
        #Make the call
        call_result = make_outbound_call(
            to_number=borrower.phone_number,
            language=normalized_language
        )
        
        if call_result.get("success"):
            result = CallResponse(
                success=True,
                call_uuid=call_result.get("call_uuid"),
                status=call_result.get("status"),
                to_number=borrower.phone_number,
                language=normalized_language,
                borrower_id=borrower.borrower_id
            )
            successful += 1
        else:
            result = CallResponse(
                success=False,
                error=call_result.get("error"),
                to_number=borrower.phone_number,
                language=normalized_language,
                borrower_id=borrower.borrower_id
            )
            failed += 1
        
        results.append(result)
    
    print(f"\n{'='*60}")
    print(f"BULK CALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Requests: {len(request.borrowers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")
    
    return BulkCallResponse(
        total_requests=len(request.borrowers),
        successful_calls=successful,
        failed_calls=failed,
        results=results
    )


@router.post("/make_call", response_model=CallResponse)
async def make_single_call(request: SingleCallRequest):
    """
    Trigger a single AI-powered call
    
    **Input:**
    - Phone number (with country code, e.g., +911234567890)
    - Preferred language (en-IN, hi-IN, ta-IN OR English, Hindi, Tamil)
    - Optional borrower ID for tracking
    
    **Output:**
    - Call UUID for tracking
    - Call status
    """
    
    # Normalize language (HINDI -> hi-IN, English -> en-IN, etc.)
    normalized_language = normalize_language(request.language)
    
    # Validate language
    if normalized_language not in settings.LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {request.language}. Supported: {list(settings.LANGUAGE_CONFIG.keys())} or English, Hindi, Tamil"
        )
    
    call_result = make_outbound_call(
        to_number=request.to_number,
        language=normalized_language
    )
    
    if call_result.get("success"):
        return CallResponse(
            success=True,
            call_uuid=call_result.get("call_uuid"),
            status=call_result.get("status"),
            to_number=request.to_number,
            language=normalized_language,
            borrower_id=request.borrower_id
        )
    else:
        return CallResponse(
            success=False,
            error=call_result.get("error"),
            to_number=request.to_number,
            language=normalized_language,
            borrower_id=request.borrower_id
        )


@router.get("/transcript/{call_uuid}")
async def get_transcript(call_uuid: str):
    """
    Get the complete transcript with AI analysis for a specific call
    
    **Returns:**
    - Conversation transcript
    - AI-generated summary
    - Sentiment analysis
    - Borrower intent classification
    - Payment date (if mentioned)
    """
    
    # Look for transcript file
    pattern = f"transcript_{call_uuid}_*.json"
    files = glob.glob(pattern)
    
    if files:
        with open(files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript not found for call UUID: {call_uuid}"
        )


@router.get("/analysis/{call_uuid}")
async def get_analysis(call_uuid: str):
    """
    Get only the AI analysis for a specific call
    
    **Returns:**
    - Summary
    - Sentiment (Positive/Neutral/Negative)
    - Intent (Paid/Will Pay/Needs Extension/Dispute/No Response)
    - Payment date (if applicable)
    """
    
    pattern = f"transcript_{call_uuid}_*.json"
    files = glob.glob(pattern)
    
    if files:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'ai_analysis' in data:
                return {
                    "call_uuid": call_uuid,
                    "ai_analysis": data['ai_analysis']
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="AI analysis not found in transcript"
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript not found for call UUID: {call_uuid}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the AI calling service
    
    **Returns:**
    - Service status
    - Active calls count
    - Supported languages
    - Available features
    """
    
    call_data_store = get_call_data_store()
    
    return {
        "status": "healthy",
        "active_calls": len([h for h in call_data_store.values() if h.is_active]),
        "total_calls": len(call_data_store),
        "sarvam_ai": "STT/TTS (saarika:v2.5 + bulbul:v2)",
        "gemini_ai": "conversation analysis" if gemini_client else "not configured",
        "supported_languages": list(settings.LANGUAGE_CONFIG.keys()),
        "features": [
            "Auto language detection",
            "Multi-language support (English, Hindi, Tamil)",
            "Real-time conversation",
            "Language switching",
            "AI-powered analysis (summary, sentiment, intent)",
            "Borrower intent classification",
            "Bulk calling support"
        ]
    }


# Note: WebSocket endpoints and webhook endpoints should be handled separately
# in a Flask app or using FastAPI WebSocket support
# The following endpoints would need to be implemented in the Flask portion:
# - /webhooks/answer (POST) - for Vonage answer webhook
# - /webhooks/event (POST) - for Vonage event webhook
# - /socket/<call_uuid> (WebSocket) - for real-time audio streaming