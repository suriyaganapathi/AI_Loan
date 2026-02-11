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
import uuid
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from app.ai_calling.service import (
    make_outbound_call,
    get_call_data_store,
    gemini_client,
    analyze_conversation_with_gemini,
    ConversationHandler
)
from config import settings


router = APIRouter()


# ============================================================
# DUMMY CONVERSATION DATA
# ============================================================

DUMMY_CONVERSATIONS = {
    "en-IN": {
        "call_uuid": "dummy-call-{uuid}",
        "start_time": "{start_time}",
        "end_time": "{end_time}",
        "duration_seconds": 52.59,
        "preferred_language": "en-IN",
        "final_language": "en-IN",
        "language_switches": 0,
        "language_history": [],
        "conversation": [
            {
                "speaker": "AI",
                "text": "Hello, I am calling from the finance agency regarding your loan payment. May I know your current payment status?",
                "timestamp": "{timestamp_1}",
                "language": "en-IN"
            },
            {
                "speaker": "User",
                "text": "I will pay day after tomorrow, is it fine?",
                "timestamp": "{timestamp_2}",
                "language": "en-IN"
            },
            {
                "speaker": "AI",
                "text": "Thank you for letting us know. Could you please confirm the specific date you expect to make the payment?",
                "timestamp": "{timestamp_3}",
                "language": "en-IN"
            },
            {
                "speaker": "User",
                "text": "Okay.",
                "timestamp": "{timestamp_4}",
                "language": "en-IN"
            },
            {
                "speaker": "AI",
                "text": "Thank you. Just to confirm, are you referring to day.",
                "timestamp": "{timestamp_5}",
                "language": "en-IN"
            },
            {
                "speaker": "User",
                "text": "I will be paying it on 12th February 2026.",
                "timestamp": "{timestamp_6}",
                "language": "en-IN"
            },
            {
                "speaker": "AI",
                "text": "Thank you for confirming the payment date. We will look forward.",
                "timestamp": "{timestamp_7}",
                "language": "en-IN"
            },
            {
                "speaker": "User",
                "text": "Okay. Okay. Thank you so much.",
                "timestamp": "{timestamp_8}",
                "language": "en-IN"
            },
            {
                "speaker": "AI",
                "text": "Thank you for your time. We appreciate you confirming the.",
                "timestamp": "{timestamp_9}",
                "language": "en-IN"
            },
            {
                "speaker": "User",
                "text": "Okay, Good Bye, Take care.",
                "timestamp": "{timestamp_10}",
                "language": "en-IN"
            },
            {
                "speaker": "AI",
                "text": "Thank you for your time. We look forward to receiving your payment on February 12, 2026.",
                "timestamp": "{timestamp_11}",
                "language": "en-IN"
            }
        ]
    },
    "hi-IN": {
        "call_uuid": "dummy-call-{uuid}",
        "start_time": "{start_time}",
        "end_time": "{end_time}",
        "duration_seconds": 45.32,
        "preferred_language": "hi-IN",
        "final_language": "hi-IN",
        "language_switches": 0,
        "language_history": [],
        "conversation": [
            {
                "speaker": "AI",
                "text": "नमस्ते, मैं वित्त एजेंसी से आपके लोन भुगतान के बारे में कॉल कर रहा हूं। कृपया अपनी वर्तमान भुगतान स्थिति बताएं?",
                "timestamp": "{timestamp_1}",
                "language": "hi-IN"
            },
            {
                "speaker": "User",
                "text": "मैं परसों पेमेंट कर दूंगा, ठीक है?",
                "timestamp": "{timestamp_2}",
                "language": "hi-IN"
            },
            {
                "speaker": "AI",
                "text": "धन्यवाद। क्या आप कृपया विशिष्ट तारीख बता सकते हैं?",
                "timestamp": "{timestamp_3}",
                "language": "hi-IN"
            },
            {
                "speaker": "User",
                "text": "मैं 12 फरवरी को भुगतान कर दूंगा।",
                "timestamp": "{timestamp_4}",
                "language": "hi-IN"
            },
            {
                "speaker": "AI",
                "text": "धन्यवाद। हम 12 फरवरी को आपके भुगतान का इंतजार करेंगे।",
                "timestamp": "{timestamp_5}",
                "language": "hi-IN"
            },
            {
                "speaker": "User",
                "text": "ठीक है, धन्यवाद।",
                "timestamp": "{timestamp_6}",
                "language": "hi-IN"
            }
        ]
    },
    "ta-IN": {
        "call_uuid": "dummy-call-{uuid}",
        "start_time": "{start_time}",
        "end_time": "{end_time}",
        "duration_seconds": 48.15,
        "preferred_language": "ta-IN",
        "final_language": "ta-IN",
        "language_switches": 0,
        "language_history": [],
        "conversation": [
            {
                "speaker": "AI",
                "text": "வணக்கம், நான் நிதி நிறுவனத்திலிருந்து உங்கள் கடன் செலுத்துதல் பற்றி அழைக்கிறேன். உங்கள் தற்போதைய கட்டண நிலையை தயவுசெய்து கூறுங்கள்?",
                "timestamp": "{timestamp_1}",
                "language": "ta-IN"
            },
            {
                "speaker": "User",
                "text": "நான் நாளை மறுநாள் செலுத்துகிறேன், சரியா?",
                "timestamp": "{timestamp_2}",
                "language": "ta-IN"
            },
            {
                "speaker": "AI",
                "text": "நன்றி. குறிப்பிட்ட தேதியை கூற முடியுமா?",
                "timestamp": "{timestamp_3}",
                "language": "ta-IN"
            },
            {
                "speaker": "User",
                "text": "நான் பிப்ரவரி 12 அன்று செலுத்துவேன்.",
                "timestamp": "{timestamp_4}",
                "language": "ta-IN"
            },
            {
                "speaker": "AI",
                "text": "நன்றி. பிப்ரவரி 12 அன்று உங்கள் கட்டணத்திற்காக காத்திருக்கிறோம்.",
                "timestamp": "{timestamp_5}",
                "language": "ta-IN"
            },
            {
                "speaker": "User",
                "text": "சரி, நன்றி.",
                "timestamp": "{timestamp_6}",
                "language": "ta-IN"
            }
        ]
    }
}


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
    
    # Internal map for common names and common typos
    language_map = {
        "HINDI": "hi-IN",
        "HIND": "hi-IN",
        "ENGLISH": "en-IN",
        "ENGISH": "en-IN",  # Common typo
        "ENGLSH": "en-IN",
        "TAMIL": "ta-IN",
        "TAML": "ta-IN",
        "EN": "en-IN",
        "HI": "hi-IN",
        "TA": "ta-IN",
        "EN-IN": "en-IN",
        "HI-IN": "hi-IN",
        "TA-IN": "ta-IN"
    }
    
    # 1. Exact map check
    if language_upper in language_map:
        return language_map[language_upper]
    
    # 2. Case-insensitive config check
    for config_key in settings.LANGUAGE_CONFIG.keys():
        if config_key.upper() == language_upper:
            return config_key
            
    # 3. Fuzzy prefix matching
    if language_upper.startswith("EN"):
        return "en-IN"
    if language_upper.startswith("HI"):
        return "hi-IN"
    if language_upper.startswith("TA"):
        return "ta-IN"
        
    return language_map.get(language_upper, language)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class BorrowerInfo(BaseModel):
    """Single borrower information"""
    NO: str = Field(..., description="Unique identifier for the borrower")
    cell1: str = Field(..., description="Phone number with country code (e.g., +911234567890)")
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
    use_dummy_data: bool = Field(default=True, description="Use dummy conversations instead of making real calls (saves credits)")
    
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
                ],
                "use_dummy_data": True
            }
        }


class SingleCallRequest(BaseModel):
    """Request model for single call"""
    to_number: str = Field(..., description="Phone number to call (with country code)")
    language: str = Field(default="en-IN", description="Preferred language: en-IN, hi-IN, or ta-IN")
    borrower_id: Optional[str] = Field(None, description="Optional borrower ID for tracking")
    use_dummy_data: bool = Field(default=True, description="Use dummy conversation instead of making real call (saves credits)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "to_number": "+911234567890",
                "language": "hi-IN",
                "borrower_id": "BRW123456",
                "use_dummy_data": True
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
    is_dummy: Optional[bool] = False
    transcript_file: Optional[str] = None
    ai_analysis: Optional[dict] = None
    conversation: Optional[List[dict]] = None


class BulkCallResponse(BaseModel):
    """Response model for bulk calling"""
    total_requests: int
    successful_calls: int
    failed_calls: int
    results: List[CallResponse]
    mode: str = "dummy"  # "dummy" or "real"


# ============================================================
# HELPER FUNCTION TO CREATE DUMMY CALL
# ============================================================

def create_dummy_call(phone_number: str, language: str, borrower_id: Optional[str] = None) -> dict:
    """
    Create a dummy call with simulated conversation and AI analysis
    This saves Vonage call credits while testing
    
    Args:
        phone_number: Phone number for the call
        language: Language code (en-IN, hi-IN, ta-IN)
        borrower_id: Optional borrower ID for tracking
    
    Returns:
        dict: Call result with UUID, status, and AI analysis
    """
    try:
        # Generate unique call UUID
        call_uuid = f"dummy-{uuid.uuid4()}"
        
        # Get dummy conversation template for the language
        if language not in DUMMY_CONVERSATIONS:
            return {
                "success": False,
                "error": f"No dummy conversation available for language: {language}"
            }
        
        # Clone the template
        conversation_template = DUMMY_CONVERSATIONS[language].copy()
        
        # Generate timestamps
        start_time = datetime.now()
        current_time = start_time
        
        # Replace placeholders in conversation
        conversation = []
        for i, entry in enumerate(conversation_template["conversation"]):
            # Add 3-8 seconds between messages
            seconds_to_add = random.uniform(3.0, 8.0)
            current_time = current_time + timedelta(seconds=seconds_to_add)
            
            conversation.append({
                "speaker": entry["speaker"],
                "text": entry["text"],
                "timestamp": current_time.isoformat(),
                "language": language
            })
        
        end_time = current_time
        duration = (end_time - start_time).total_seconds()
        
        # Perform AI analysis on the conversation
        print(f"[DUMMY CALL] 🤖 Running AI analysis for call {call_uuid} (Borrower: {borrower_id})")
        ai_analysis = analyze_conversation_with_gemini(conversation)
        
        # Create transcript data
        transcript_data = {
            "call_uuid": call_uuid,
            "borrower_id": borrower_id,
            "phone_number": phone_number,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "preferred_language": language,
            "final_language": language,
            "language_switches": 0,
            "language_history": [],
            "conversation": conversation,
            "ai_analysis": ai_analysis,
            "is_dummy": True,
            "note": "This is a simulated conversation for testing purposes. No actual call was made."
        }
        
        # Save transcript to file (using hidden folder to avoid triggering frontend reloads)
        import os
        os.makedirs(".transcripts", exist_ok=True)
        filename = f".transcripts/transcript_{call_uuid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        print(f"[DUMMY CALL] ✅ Transcript saved: {filename}")
        
        # Print AI analysis summary
        if ai_analysis:
            print(f"  📊 Sentiment: {ai_analysis.get('sentiment', 'N/A')} | Intent: {ai_analysis.get('intent', 'N/A')}")
        
        return {
            "success": True,
            "call_uuid": call_uuid,
            "status": "completed (dummy)",
            "to_number": phone_number,
            "language": language,
            "borrower_id": borrower_id,
            "is_dummy": True,
            "transcript_file": filename,
            "ai_analysis": ai_analysis,
            "conversation": conversation
        }
        
    except Exception as e:
        print(f"[DUMMY CALL] ❌ Error creating dummy call for {borrower_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def process_single_call(borrower: BorrowerInfo, use_dummy_data: bool, normalized_language: str) -> CallResponse:
    """
    Process a single call (dummy or real) for parallel execution
    
    Args:
        borrower: Borrower information
        use_dummy_data: Whether to use dummy data or make real call
        normalized_language: Normalized language code
    
    Returns:
        CallResponse: Result of the call
    """
    print(f"[PARALLEL] Processing borrower: {borrower.NO} ({normalized_language})")
    
    # Make the call (dummy or real)
    if use_dummy_data:
        call_result = create_dummy_call(
            phone_number=borrower.cell1,
            language=normalized_language,
            borrower_id=borrower.NO
        )
    else:
        call_result = make_outbound_call(
            to_number=borrower.cell1,
            language=normalized_language
        )
    
    # Create response
    if call_result.get("success"):
        return CallResponse(
            success=True,
            call_uuid=call_result.get("call_uuid"),
            status=call_result.get("status"),
            to_number=borrower.cell1,
            language=normalized_language,
            borrower_id=borrower.NO,
            is_dummy=use_dummy_data,
            transcript_file=call_result.get("transcript_file"),
            ai_analysis=call_result.get("ai_analysis"),
            conversation=call_result.get("conversation")
        )
    else:
        return CallResponse(
            success=False,
            error=call_result.get("error"),
            to_number=borrower.cell1,
            language=normalized_language,
            borrower_id=borrower.NO,
            is_dummy=use_dummy_data
        )


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
            "Intent classification",
            "Dummy mode (saves call credits during testing)"
        ],
        "modes": {
            "dummy": "Use simulated conversations with AI analysis (no actual calls made)",
            "real": "Make actual calls via Vonage (uses call credits)"
        }
    }


@router.post("/trigger_calls", response_model=BulkCallResponse)
async def trigger_bulk_calls(request: BulkCallRequest):
    """
    Trigger multiple calls to borrowers with their preferred languages using PARALLEL PROCESSING
    
    This endpoint allows you to initiate multiple AI-powered calls in one request.
    Each borrower can have a different preferred language (English, Hindi, or Tamil).
    
    **Performance:**
    - Uses parallel processing with ThreadPoolExecutor
    - Processes multiple calls simultaneously instead of sequentially
    - Significantly faster for large batches
    
    **Modes:**
    - **Dummy Mode** (use_dummy_data=true): Uses simulated conversations with AI analysis. No actual calls made. Saves credits.
    - **Real Mode** (use_dummy_data=false): Makes actual calls via Vonage. Uses call credits.
    
    **Input:**
    - List of borrowers with their IDs, phone numbers, and preferred languages
    - use_dummy_data flag (default: true)
    
    **Output:**
    - Summary of successful and failed calls
    - Individual call UUIDs for tracking
    - AI analysis results (summary, sentiment, intent)
    - Processing time statistics
    
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
    
    mode = "DUMMY" if request.use_dummy_data else "REAL"
    total_borrowers = len(request.borrowers)
    
    print(f"\n{'='*60}")
    print(f"📞 BULK CALL REQUEST ({mode} MODE) - {total_borrowers} borrowers")
    print(f"🚀 Using PARALLEL PROCESSING for faster execution")
    print(f"{'='*60}\n")
    
    # Start timing
    start_time = datetime.now()
    
    # Prepare tasks for parallel processing
    tasks = []
    
    for borrower in request.borrowers:
        # Normalize language (HINDI -> hi-IN, English -> en-IN, etc.)
        normalized_language = normalize_language(borrower.preferred_language)
        
        # Validate language
        if normalized_language not in settings.LANGUAGE_CONFIG:
            # Add failed result immediately for invalid language
            tasks.append({
                "borrower": borrower,
                "normalized_language": normalized_language,
                "valid": False,
                "error": f"Unsupported language: {borrower.preferred_language}"
            })
        else:
            tasks.append({
                "borrower": borrower,
                "normalized_language": normalized_language,
                "valid": True
            })
    
    # Process calls in parallel using ThreadPoolExecutor
    results = []
    successful = 0
    failed = 0
    
    # Determine optimal number of workers (max 10 concurrent calls)
    max_workers = min(10, total_borrowers)
    
    print(f"⚙️  Processing with {max_workers} parallel workers...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all valid tasks
        future_to_task = {}
        
        for task in tasks:
            if not task["valid"]:
                # Add failed result for invalid tasks
                result = CallResponse(
                    success=False,
                    error=task["error"],
                    to_number=task["borrower"].cell1,
                    borrower_id=task["borrower"].NO,
                    language=task["borrower"].preferred_language,
                    is_dummy=request.use_dummy_data
                )
                results.append(result)
                failed += 1
            else:
                # Submit valid task for parallel execution
                future = executor.submit(
                    process_single_call,
                    task["borrower"],
                    request.use_dummy_data,
                    task["normalized_language"]
                )
                future_to_task[future] = task
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"[ERROR] Exception processing borrower {task['borrower'].NO}: {e}")
                result = CallResponse(
                    success=False,
                    error=f"Processing error: {str(e)}",
                    to_number=task["borrower"].cell1,
                    language=task["normalized_language"],
                    borrower_id=task["borrower"].NO,
                    is_dummy=request.use_dummy_data
                )
                results.append(result)
                failed += 1
    
    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print(f"BULK CALL SUMMARY ({mode} MODE)")
    print(f"{'='*60}")
    print(f"Total Requests: {total_borrowers}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Average Time per Call: {processing_time/total_borrowers:.2f} seconds")
    print(f"{'='*60}\n")
    
    return BulkCallResponse(
        total_requests=total_borrowers,
        successful_calls=successful,
        failed_calls=failed,
        results=results,
        mode="dummy" if request.use_dummy_data else "real"
    )


@router.post("/make_call", response_model=CallResponse)
async def make_single_call(request: SingleCallRequest):
    """
    Trigger a single AI-powered call
    
    **Modes:**
    - **Dummy Mode** (use_dummy_data=true): Uses simulated conversation with AI analysis. No actual call made.
    - **Real Mode** (use_dummy_data=false): Makes actual call via Vonage. Uses call credits.
    
    **Input:**
    - Phone number (with country code, e.g., +911234567890)
    - Preferred language (en-IN, hi-IN, ta-IN OR English, Hindi, Tamil)
    - Optional borrower ID for tracking
    - use_dummy_data flag (default: true)
    
    **Output:**
    - Call UUID for tracking
    - Call status
    - AI analysis (if dummy mode)
    - Transcript file location
    """
    
    # Normalize language (HINDI -> hi-IN, English -> en-IN, etc.)
    normalized_language = normalize_language(request.language)
    
    # Validate language
    if normalized_language not in settings.LANGUAGE_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {request.language}. Supported: {list(settings.LANGUAGE_CONFIG.keys())} or English, Hindi, Tamil"
        )
    
    # Make the call (dummy or real)
    if request.use_dummy_data:
        print(f"[SINGLE CALL] Mode: DUMMY (no actual call)")
        call_result = create_dummy_call(
            phone_number=request.to_number,
            language=normalized_language,
            borrower_id=request.borrower_id
        )
    else:
        print(f"[SINGLE CALL] Mode: REAL (making actual call)")
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
            borrower_id=request.borrower_id,
            is_dummy=request.use_dummy_data,
            transcript_file=call_result.get("transcript_file"),
            ai_analysis=call_result.get("ai_analysis"),
            conversation=call_result.get("conversation")
        )
    else:
        return CallResponse(
            success=False,
            error=call_result.get("error"),
            to_number=request.to_number,
            language=normalized_language,
            borrower_id=request.borrower_id,
            is_dummy=request.use_dummy_data
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
    - Dummy flag (indicates if this was a simulated call)
    """
    
    # Look for transcript file
    pattern = f".transcripts/transcript_{call_uuid}_*.json"
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
    - Dummy flag (indicates if this was a simulated call)
    """
    
    pattern = f".transcripts/transcript_{call_uuid}_*.json"
    files = glob.glob(pattern)
    
    if files:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'ai_analysis' in data:
                return {
                    "call_uuid": call_uuid,
                    "is_dummy": data.get("is_dummy", False),
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
    - Available modes (dummy/real)
    """
    
    call_data_store = get_call_data_store()
    
    return {
        "status": "healthy",
        "active_calls": len([h for h in call_data_store.values() if h.is_active]),
        "total_calls": len(call_data_store),
        "sarvam_ai": "STT/TTS (saarika:v2.5 + bulbul:v2)",
        "gemini_ai": "conversation analysis" if gemini_client else "not configured",
        "supported_languages": list(settings.LANGUAGE_CONFIG.keys()),
        "modes": {
            "dummy": "Simulated conversations with AI analysis (no call credits used)",
            "real": "Actual phone calls via Vonage (uses call credits)"
        },
        "features": [
            "Auto language detection",
            "Multi-language support (English, Hindi, Tamil)",
            "Real-time conversation",
            "Language switching",
            "AI-powered analysis (summary, sentiment, intent)",
            "Borrower intent classification",
            "Bulk calling support",
            "Dummy mode for testing"
        ]
    }


# Note: WebSocket endpoints and webhook endpoints should be handled separately
# in a Flask app or using FastAPI WebSocket support
# The following endpoints would need to be implemented in the Flask portion:
# - /webhooks/answer (POST) - for Vonage answer webhook
# - /webhooks/event (POST) - for Vonage event webhook
# - /socket/<call_uuid> (WebSocket) - for real-time audio streaming