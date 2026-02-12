"""
AI Calling Service - COMPLETE WORKING VERSION
==================
Core service for handling AI-powered phone calls using Vonage, Sarvam AI, and Gemini
"""

import os
import json
import base64
import uuid
import time
import jwt
import wave
import struct
import threading
from io import BytesIO
from datetime import datetime
from queue import Queue
import re

import requests
from vonage import Vonage, Auth

# Import Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: google-genai not installed. Install with: pip install google-genai")
    GEMINI_AVAILABLE = False

from config import settings


# ============================================================
# GLOBAL STORAGE
# ============================================================

call_data = {}
audio_cache = {}

# Initialize Vonage client
try:
    vonage_client = Vonage(Auth(
        application_id=settings.VONAGE_APPLICATION_ID,
        private_key=settings.VONAGE_PRIVATE_KEY_PATH
    ))
    voice = vonage_client.voice
    print("[VONAGE] ✅ Vonage Voice client initialized")
except Exception as e:
    print(f"[VONAGE] ⚠️  Failed to initialize: {e}")
    vonage_client = None
    voice = None

# Initialize Gemini AI client
gemini_client = None
if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        print("[GEMINI] ✅ Gemini AI client initialized")
    except Exception as e:
        print(f"[GEMINI] ⚠️  Failed to initialize: {e}")
        gemini_client = None
else:
    print("[GEMINI] ⚠️  Gemini not configured - AI analysis will be disabled")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_jwt_token():
    """Generate JWT token for Vonage API"""
    try:
        with open(settings.VONAGE_PRIVATE_KEY_PATH, 'rb') as key_file:
            private_key = key_file.read()
        
        payload = {
            'application_id': settings.VONAGE_APPLICATION_ID,
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'jti': str(uuid.uuid4())
        }
        
        return jwt.encode(payload, private_key, algorithm='RS256')
    except Exception as e:
        print(f"[JWT] Error: {e}")
        return None


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def analyze_conversation_with_gemini(conversation):
    """
    Analyze conversation using Gemini AI to extract:
    1. Summary of conversation
    2. Sentiment (Positive/Neutral/Negative)
    3. Borrower Intent (Paid/Will Pay/Needs Extension/Dispute/No Response)
    """
    
    if not gemini_client:
        print("[GEMINI] ⚠️  Gemini client not available, skipping analysis")
        return {
            "summary": "AI analysis not available - Gemini API not configured",
            "sentiment": "Neutral",
            "sentiment_reasoning": "Analysis skipped",
            "intent": "No Response",
            "intent_reasoning": "Analysis skipped",
            "payment_date": None
        }
    
    # Prepare conversation text
    conversation_text = "\n".join([
        f"{entry['speaker']}: {entry['text']}" 
        for entry in conversation
    ])
    
    prompt = f"""You are an AI analyst reviewing a phone conversation between a collection agent (AI) and a borrower (User).

Analyze this conversation and provide:

1. **SUMMARY**: A concise 2-3 sentence summary of what was discussed in the conversation.

2. **SENTIMENT**: Classify the borrower's overall sentiment as one of:
   - Positive (cooperative, friendly, willing to resolve)
   - Neutral (matter-of-fact, neither positive nor negative)
   - Negative (angry, frustrated, hostile, uncooperative)

3. **INTENT**: Classify the borrower's intent as ONE of:
   - Paid (already made payment or claims to have paid)
   - Will Pay (committed to making payment, provide the date if mentioned)
   - Needs Extension (requesting more time or a payment plan)
   - Dispute (disputing the debt or claiming error)
   - No Response (minimal engagement, evasive, or hung up quickly)

CONVERSATION:
{conversation_text}

Respond in JSON format only:
{{
    "summary": "Brief summary of the conversation",
    "sentiment": "Positive|Neutral|Negative",
    "sentiment_reasoning": "Brief explanation of why you chose this sentiment",
    "intent": "Paid|Will Pay|Needs Extension|Dispute|No Response",
    "intent_reasoning": "Brief explanation of why you chose this intent",
    "payment_date": "YYYY-MM-DD or null if not mentioned or not applicable"
}}"""
    
    try:
        print(f"\n[GEMINI] 🤖 Starting AI analysis...")
        
        # Use the new Gemini SDK
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        analysis = json.loads(response_text)
        
        print(f"[GEMINI] ✅ Analysis completed successfully")
        print(f"[GEMINI] 📊 Sentiment: {analysis.get('sentiment')}")
        print(f"[GEMINI] 🎯 Intent: {analysis.get('intent')}")
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"[GEMINI] ❌ JSON parsing error: {e}")
        print(f"[GEMINI] Response text: {response_text[:200]}")
        
        return {
            "summary": "Unable to analyze conversation - parsing error",
            "sentiment": "Neutral",
            "sentiment_reasoning": "Error in analysis",
            "intent": "No Response",
            "intent_reasoning": "Error in analysis",
            "payment_date": None,
            "error": str(e)
        }
        
    except Exception as e:
        print(f"[GEMINI] ❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "summary": "Unable to analyze conversation - API error",
            "sentiment": "Neutral",
            "sentiment_reasoning": "Error in analysis",
            "intent": "No Response",
            "intent_reasoning": "Error in analysis",
            "payment_date": None,
            "error": str(e)
        }


# ============================================================
# SARVAM AI - STT/TTS
# ============================================================

def transcribe_sarvam(audio_data, language="en-IN", max_retries=2):
    """Transcribe audio using Sarvam AI STT (saarika:v2.5) with retry logic"""
    
    # Skip if audio is too short (less than 0.3 seconds)
    min_audio_size = settings.SAMPLE_RATE * settings.SAMPLE_WIDTH * 0.3
    if len(audio_data) < min_audio_size:
        print(f"[STT] ⚠️  Audio too short ({len(audio_data)} bytes), skipping")
        return None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[STT] 🔄 Retry attempt {attempt + 1}/{max_retries}")
            
            print(f"[STT] 🎤 Transcribing audio ({len(audio_data)} bytes, {language})...")
            
            # Convert raw PCM audio to WAV format
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(settings.CHANNELS)  # Mono
                wav_file.setsampwidth(settings.SAMPLE_WIDTH)  # 16-bit
                wav_file.setframerate(settings.SAMPLE_RATE)  # 16kHz
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            
            # Prepare multipart form data
            headers = {
                'api-subscription-key': settings.SARVAM_API_KEY,
            }
            
            files = {
                'file': ('audio.wav', wav_buffer, 'audio/wav')
            }
            
            data = {
                'language_code': language,
                'model': 'saarika:v2.5'
            }
            
            # Reduced timeout to 10 seconds for faster failure
            response = requests.post(
                'https://api.sarvam.ai/speech-to-text',
                headers=headers,
                files=files,
                data=data,
                timeout=10  # Reduced from 30 to 10 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('transcript', '')
                
                if transcript:
                    print(f"[STT] ✅ Transcribed: '{transcript}'")
                    return transcript
                else:
                    print("[STT] ⚠️  Empty transcript")
                    return None
            else:
                print(f"[STT] ❌ API Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief pause before retry
                    continue
                return None
                
        except requests.exceptions.Timeout:
            print(f"[STT] ⏱️  Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            print("[STT] ❌ All retry attempts failed due to timeout")
            return None
            
        except Exception as e:
            print(f"[STT] ❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            import traceback
            traceback.print_exc()
            return None
    
    return None


def synthesize_sarvam(text, language="en-IN", max_retries=2):
    """Convert text to speech using Sarvam AI TTS (bulbul:v2) with retry logic"""
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[TTS] 🔄 Retry attempt {attempt + 1}/{max_retries}")
            
            # Get speaker and preprocessing from config
            config = settings.LANGUAGE_CONFIG.get(language, {})
            speaker = config.get('speaker', 'manisha')
            enable_preprocessing = config.get('enable_preprocessing', False)
            
            headers = {
                'api-subscription-key': settings.SARVAM_API_KEY,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'inputs': [text],
                'target_language_code': language,
                'speaker': speaker,
                'pitch': 0,
                'pace': 1.0,
                'loudness': 1.5,
                'speech_sample_rate': 16000,
                'enable_preprocessing': enable_preprocessing,
                'model': 'bulbul:v2'
            }
            
            print(f"[TTS] 🔊 Synthesizing: '{text[:50]}...' ({language}, {speaker})")
            
            # Reduced timeout to 10 seconds
            response = requests.post(
                'https://api.sarvam.ai/text-to-speech',
                headers=headers,
                json=payload,
                timeout=10  # Reduced from 30 to 10 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                audios = result.get('audios', [])
                
                if audios and audios[0]:
                    audio_base64 = audios[0]
                    audio_bytes = base64.b64decode(audio_base64)
                    print(f"[TTS] ✅ Generated {len(audio_bytes)} bytes of audio")
                    return audio_bytes
                else:
                    print("[TTS] ⚠️  No audio in response")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return None
            else:
                print(f"[TTS] ❌ API Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
                
        except requests.exceptions.Timeout:
            print(f"[TTS] ⏱️  Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            print("[TTS] ❌ All retry attempts failed due to timeout")
            return None
            
        except Exception as e:
            print(f"[TTS] ❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            import traceback
            traceback.print_exc()
            return None
    
    return None


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    """Simple language detection based on character sets"""
    text = text.strip()
    
    # Check for Devanagari script (Hindi)
    if re.search(r'[\u0900-\u097F]', text):
        return "hi-IN"
    
    # Check for Tamil script
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta-IN"
    
    # Default to English
    return "en-IN"


# ============================================================
# AUDIO BUFFERING
# ============================================================

class AudioBuffer:
    """Buffer audio chunks and detect silence"""
    
    def __init__(self, silence_threshold=500, silence_duration=1.8):  # Increased from 1.0 to 1.8 for better conversation flow
        self.buffer = BytesIO()
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.silence_start = None
        self.sample_rate = settings.SAMPLE_RATE
        self.last_chunk_time = time.time()
        self.speech_detected = False  # Track if we've detected speech
        self.min_speech_duration = 1.5  # Minimum 1.5 seconds of audio before processing
        
    def add_chunk(self, audio_chunk):
        """Add audio chunk and detect if ready to process"""
        self.buffer.write(audio_chunk)
        current_time = time.time()
        
        # Calculate RMS volume
        try:
            samples = struct.unpack(f'{len(audio_chunk)//2}h', audio_chunk)
            rms = sum(abs(s) for s in samples) / len(samples) if samples else 0
        except:
            rms = 0
        
        # Detect if speech has started
        if rms >= self.silence_threshold:
            self.speech_detected = True
            self.silence_start = None  # Reset silence counter when speech is detected
        
        # Only check for silence AFTER speech has been detected
        if self.speech_detected and rms < self.silence_threshold:
            if self.silence_start is None:
                self.silence_start = current_time
            elif current_time - self.silence_start >= self.silence_duration:
                # Silence detected for required duration after speech
                # Ensure we have at least 1.5 seconds of audio
                min_buffer_size = int(self.sample_rate * 2 * self.min_speech_duration)
                if self.buffer.tell() > min_buffer_size:
                    return True
        
        # Process if buffer gets too large (8 seconds max to allow longer utterances)
        max_buffer_size = settings.SAMPLE_RATE * 2 * 8  # 8 seconds max
        if self.buffer.tell() > max_buffer_size:
            # Only process if we've detected speech
            if self.speech_detected:
                return True
        
        self.last_chunk_time = current_time
        return False
    
    def get_audio(self):
        """Get buffered audio and reset"""
        audio_data = self.buffer.getvalue()
        self.buffer = BytesIO()
        self.silence_start = None
        self.speech_detected = False  # Reset speech detection for next utterance
        return audio_data


# ============================================================
# AI RESPONSE GENERATION
# ============================================================

def generate_ai_response(user_text, language="en-IN", context=None):
    """
    Generate AI response based on user input and language using Gemini AI
    Focused on finance collection calls with specific intent capture
    """
    
    if not gemini_client:
        print("[AI RESPONSE] ⚠️  Gemini client not available, using fallback")
        user_lower = user_text.lower()
        # Route to language-specific fallback responses
        if language == "hi-IN":
            return generate_hindi_response(user_lower)
        elif language == "ta-IN":
            return generate_tamil_response(user_lower)
        else:
            return generate_english_response(user_lower)
    
    # Get language configuration
    lang_config = settings.LANGUAGE_CONFIG.get(language, settings.LANGUAGE_CONFIG["en-IN"])
    lang_name = lang_config["name"]
    
    # Build conversation history from context
    conversation_history = ""
    if context and "conversation" in context and context["conversation"]:
        conversation_history = "\n".join([
            f"{entry['speaker']}: {entry['text']}" 
            for entry in context["conversation"][-5:]  # Last 5 exchanges for context
        ])
    
    # Create dynamic prompt for Gemini based on language
    if language == "en-IN":
        system_prompt = """You are an automated assistant calling on behalf of a finance agency for loan collection purposes.

Your conversation guidelines:
1. Be polite, professional, and compliant with collection regulations
2. Keep responses SHORT and CONVERSATIONAL (1-2 sentences max)
3. Focus ONLY on finance-related matters (loan payments, EMI, outstanding amounts)
4. Your goal is to understand the borrower's payment status and intent

Conversation flow:
- If this is the first interaction: Introduce yourself clearly as an automated assistant from the finance agency
- Ask about their payment status if not yet discussed
- Capture borrower intent through natural conversation:
  a) Already Paid - they claim payment is complete
  b) Will Pay - they commit to paying (try to get a specific date)
  c) Needs Extension - they request more time or payment plan
  d) Dispute - they dispute the debt or claim there's an error
  e) No clear response - they're evasive or unclear

Keep the call SHORT and focused. Do NOT discuss unrelated topics. If they ask about non-finance matters, politely redirect to the payment discussion.

Respond in English only."""
    
    elif language == "hi-IN":
        system_prompt = """आप एक वित्त एजेंसी की ओर से लोन वसूली के लिए कॉल करने वाले स्वचालित सहायक हैं।

आपके वार्तालाप दिशानिर्देश:
1. विनम्र, पेशेवर और संग्रह नियमों के अनुरूप रहें
2. जवाब छोटे और संवादात्मक रखें
3. केवल वित्त से संबंधित मामलों पर ध्यान दें (लोन भुगतान, EMI, बकाया राशि)
4. आपका लक्ष्य उधारकर्ता की भुगतान स्थिति और इरादे को समझना है

वार्तालाप प्रवाह:
- यदि यह पहली बातचीत है: वित्त एजेंसी से स्वचालित सहायक के रूप में अपना स्पष्ट परिचय दें
- उनकी भुगतान स्थिति के बारे में पूछें यदि अभी तक चर्चा नहीं हुई है
- प्राकृतिक बातचीत के माध्यम से उधारकर्ता के इरादे को पकड़ें:
  a) पहले ही भुगतान कर दिया - वे दावा करते हैं कि भुगतान पूरा हो गया है
  b) भुगतान करेंगे - वे भुगतान करने के लिए प्रतिबद्ध हैं (एक विशिष्ट तारीख प्राप्त करने का प्रयास करें)
  c) विस्तार चाहिए - वे अधिक समय या भुगतान योजना का अनुरोध करते हैं
  d) विवाद - वे ऋण पर विवाद करते हैं या त्रुटि का दावा करते हैं
  e) कोई स्पष्ट प्रतिक्रिया नहीं - वे टालमटोल करते हैं या अस्पष्ट हैं

कॉल को छोटा और केंद्रित रखें। असंबंधित विषयों पर चर्चा न करें। यदि वे गैर-वित्त मामलों के बारे में पूछते हैं, तो विनम्रता से भुगतान चर्चा की ओर पुनर्निर्देशित करें।

केवल हिंदी में जवाब दें।"""
    
    else:  # Tamil (ta-IN)
        system_prompt = """நீங்கள் கடன் வசூலுக்காக நிதி நிறுவனத்தின் சார்பாக அழைக்கும் தானியங்கி உதவியாளர்.

உங்கள் உரையாடல் வழிகாட்டுதல்கள்:
1. கண்ணியமாக, தொழில்முறையாக மற்றும் சேகரிப்பு விதிமுறைகளுக்கு இணங்க இருங்கள்
2. பதில்களை குறுகியதாகவும் உரையாடல் முறையிலும் வைத்திருங்கள்
3. நிதி தொடர்பான விஷயங்களில் மட்டுமே கவனம் செலுத்துங்கள் (கடன் செலுத்துதல், EMI, நிலுவைத் தொகை)
4. உங்கள் குறிக்கோள் கடன் வாங்கியவரின் கட்டண நிலை மற்றும் நோக்கத்தை புரிந்து கொள்வது

உரையாடல் ஓட்டம்:
- இது முதல் தொடர்பு என்றால்: நிதி நிறுவனத்தின் தானியங்கி உதவியாளராக உங்களை தெளிவாக அறிமுகப்படுத்துங்கள்
- இன்னும் விவாதிக்கப்படவில்லை என்றால் அவர்களின் கட்டண நிலையைப் பற்றி கேளுங்கள்
- இயல்பான உரையாடல் மூலம் கடன் வாங்கியவரின் நோக்கத்தை கண்டறியுங்கள்:
  a) ஏற்கனவே செலுத்தியது - அவர்கள் கட்டணம் முடிந்துவிட்டது என்று கூறுகிறார்கள்
  b) செலுத்துவார்கள் - அவர்கள் செலுத்த உறுதியளிக்கிறார்கள் (குறிப்பிட்ட தேதியைப் பெற முயற்சிக்கவும்)
  c) நீட்டிப்பு தேவை - அவர்கள் அதிக நேரம் அல்லது கட்டணத் திட்டத்தைக் கோருகிறார்கள்
  d) சர்ச்சை - அவர்கள் கடனைப் பற்றி சர்ச்சை செய்கிறார்கள் அல்லது பிழை இருப்பதாகக் கூறுகிறார்கள்
  e) தெளிவான பதில் இல்லை - அவர்கள் தவிர்க்கிறார்கள் அல்லது தெளிவற்றவர்கள்

அழைப்பைக் குறுகியதாகவும் கவனம் செலுத்துவதாகவும் வைத்திருங்கள். தொடர்பில்லாத தலைப்புகளை விவாதிக்க வேண்டாம். அவர்கள் நிதி அல்லாத விஷயங்களைப் பற்றி கேட்டால், கண்ணியமாக கட்டண விவாதத்திற்கு திருப்பி விடுங்கள்.

தமிழில் மட்டும் பதிலளியுங்கள்."""
    
    # Create the full prompt with conversation context
    prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{conversation_history if conversation_history else "This is the start of the conversation."}

USER'S LATEST MESSAGE: {user_text}

Generate a natural, conversational response in {lang_name}. Keep it brief but ALWAYS complete your sentences. Respond in 1-2 complete sentences that are focused on understanding their payment status or moving the conversation forward. Make sure your response ends with proper punctuation."""
    
    try:
        print(f"[AI RESPONSE] 🤖 Generating response using Gemini AI ({lang_name})...")
        
        # Call Gemini API
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=300,  # Increased from 150 to 300 to ensure complete sentences
            )
        )
        
        ai_response = response.text.strip()
        
        # Ensure the response ends with proper punctuation
        # This helps avoid cut-off sentences
        if ai_response and not ai_response[-1] in ['.', '?', '!', '।', '॥']:
            # If response doesn't end with punctuation, add a period
            if language == "hi-IN":
                ai_response += "।"  # Hindi full stop
            elif language == "ta-IN":
                ai_response += "."  # Tamil uses period
            else:
                ai_response += "."  # English period
        
        print(f"[AI RESPONSE] ✅ Generated: {ai_response}")
        
        return ai_response
        
    except Exception as e:
        print(f"[AI RESPONSE] ❌ Gemini API error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# CONVERSATION HANDLER
# ============================================================

class ConversationHandler:
    """Manages conversation state and transcript"""
    
    def __init__(self, call_uuid, preferred_language="en-IN", borrower_id=None):
        self.call_uuid = call_uuid
        self.conversation = []
        self.context = {}
        self.is_active = True
        self.start_time = datetime.now()
        self.preferred_language = preferred_language  # Store preferred language
        self.current_language = preferred_language    # Start with preferred language
        self.language_history = []
        self.borrower_id = borrower_id # Store borrower ID for updates
        
    def add_entry(self, speaker, text):
        """Add conversation entry"""
        entry = {
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "language": self.current_language
        }
        self.conversation.append(entry)
        # Update context with conversation for AI response generation
        self.context["conversation"] = self.conversation
        print(f"[CONV] [{speaker}] [{self.current_language}] {text}")
    
    def update_language(self, detected_language):
        """Update conversation language"""
        if detected_language != self.current_language:
            old_lang = settings.LANGUAGE_CONFIG.get(self.current_language, {}).get("name", self.current_language)
            new_lang = settings.LANGUAGE_CONFIG.get(detected_language, {}).get("name", detected_language)
            print(f"[LANG] 🔄 Switching from {old_lang} to {new_lang}")
            
            self.language_history.append({
                "from": self.current_language,
                "to": detected_language,
                "timestamp": datetime.now().isoformat()
            })
            
            self.current_language = detected_language
    
    def save_transcript(self):
        """Save conversation transcript with AI analysis"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        ai_analysis = None
        if len(self.conversation) > 1:
            print(f"\n[AI ANALYSIS] Starting Gemini AI analysis for call {self.call_uuid}")
            ai_analysis = analyze_conversation_with_gemini(self.conversation)
        else:
            print(f"[AI ANALYSIS] Skipping analysis - insufficient conversation data")
            ai_analysis = {
                "summary": "No meaningful conversation detected",
                "sentiment": "No Response",
                "sentiment_reasoning": "Insufficient data",
                "intent": "No Response",
                "intent_reasoning": "Call ended without engagement",
                "payment_date": None
            }
        
        import os
        os.makedirs(".transcripts", exist_ok=True)
        filename = f".transcripts/transcript_{self.call_uuid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        transcript_data = {
            "call_uuid": self.call_uuid,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "preferred_language": self.preferred_language,
            "final_language": self.current_language,
            "language_switches": len(self.language_history),
            "language_history": self.language_history,
            "conversation": self.conversation,
            "ai_analysis": ai_analysis
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        if ai_analysis:
            print(f"\n{'='*60}")
            print(f"AI ANALYSIS SUMMARY - {self.call_uuid}")
            print(f"{'='*60}")
            print(f"📝 Summary: {ai_analysis.get('summary', 'N/A')}")
            print(f"😊 Sentiment: {ai_analysis.get('sentiment', 'N/A')} - {ai_analysis.get('sentiment_reasoning', 'N/A')}")
            print(f"🎯 Intent: {ai_analysis.get('intent', 'N/A')} - {ai_analysis.get('intent_reasoning', 'N/A')}")
            if ai_analysis.get('payment_date'):
                print(f"📅 Payment Date: {ai_analysis.get('payment_date')}")
            print(f"{'='*60}\n")
        
        return filename, ai_analysis


# ============================================================
# CALL MANAGEMENT
# ============================================================

def make_outbound_call(to_number, language="en-IN", borrower_id=None):
    """Trigger an outbound call with preferred language"""
    if not voice:
        return {"success": False, "error": "Vonage client not initialized"}
    
    # Strip '+' for Vonage SDK
    if to_number.startswith('+'):
        to_number = to_number[1:]
    
    try:
        # Create call with language parameter in answer URL
        answer_url = f'{settings.BASE_URL}/webhooks/answer?preferred_language={language}'
        
        if borrower_id:
            answer_url += f"&borrower_id={borrower_id}"
        
        response = voice.create_call({
            'to': [{'type': 'phone', 'number': to_number}],
            'from_': {'type': 'phone', 'number': settings.VONAGE_FROM_NUMBER},
            'answer_url': [answer_url],
            'event_url': [f'{settings.BASE_URL}/webhooks/event']
        })
        
        call_uuid = response.uuid
        
        print(f"\n{'*'*60}")
        print(f"📞 OUTBOUND CALL INITIATED")
        print(f"{'*'*60}")
        print(f"To: {to_number}")
        print(f"UUID: {call_uuid}")
        print(f"Preferred Language: {language}")
        print(f"Borrower ID: {borrower_id}")
        print(f"Answer URL: {answer_url}")
        print(f"Event URL: {settings.BASE_URL}/webhooks/event")
        print(f"{'*'*60}\n")
        
        return {
            "success": True,
            "call_uuid": call_uuid,
            "status": getattr(response, 'status', 'initiated'),
            "to_number": to_number,
            "language": language,
            "borrower_id": borrower_id
        }
        
    except Exception as e:
        print(f"[ERROR] ❌ {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_call_data_store():
    """Get the global call data storage"""
    return call_data