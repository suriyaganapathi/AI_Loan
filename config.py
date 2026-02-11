"""
Application Configuration
=========================
Central configuration for all environment variables and settings
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application Settings"""
    
    # ---------- Gemini AI ----------
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Alternative name
    
    # ---------- Vonage Configuration ----------
    VONAGE_API_KEY = os.getenv("VONAGE_API_KEY", "")
    VONAGE_API_SECRET = os.getenv("VONAGE_API_SECRET", "")
    VONAGE_APPLICATION_ID = os.getenv("VONAGE_APPLICATION_ID", "")
    VONAGE_PRIVATE_KEY_PATH = os.getenv("VONAGE_PRIVATE_KEY_PATH", "private.key")
    VONAGE_FROM_NUMBER = os.getenv("VONAGE_FROM_NUMBER", "")
    
    # ---------- Sarvam AI Configuration ----------
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
    
    # ---------- Server Configuration ----------
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    
    # ---------- Audio Configuration ----------
    SAMPLE_RATE = 16000  # Vonage WebSocket uses 16kHz
    CHANNELS = 1         # Mono
    SAMPLE_WIDTH = 2     # 16-bit
    
    # ---------- Language Configuration ----------
    LANGUAGE_CONFIG = {
        "en-IN": {
            "name": "English",
            "speaker": "manisha",
            "enable_preprocessing": False,
            "greeting": "Hello, I am calling from the finance agency regarding your loan payment. May I know your current payment status?"
        },
        "hi-IN": {
            "name": "Hindi",
            "speaker": "manisha",
            "enable_preprocessing": True,
            "greeting": "नमस्ते, मैं वित्त एजेंसी से आपके लोन भुगतान के बारे में कॉल कर रहा हूं। कृपया अपनी वर्तमान भुगतान स्थिति बताएं?"
        },
        "ta-IN": {
            "name": "Tamil",
            "speaker": "manisha",
            "enable_preprocessing": True,
            "greeting": "வணக்கம், நான் நிதி நிறுவனத்திலிருந்து உங்கள் கடன் செலுத்துதல் பற்றி அழைக்கிறேன். உங்கள் தற்போதைய கட்டண நிலையை தயவுசெய்து கூறுங்கள்?"
        }
    }
    
    @classmethod
    def validate(cls):
        """Validate that required settings are present"""
        required_settings = {
            "VONAGE_API_KEY": cls.VONAGE_API_KEY,
            "VONAGE_API_SECRET": cls.VONAGE_API_SECRET,
            "VONAGE_APPLICATION_ID": cls.VONAGE_APPLICATION_ID,
            "SARVAM_API_KEY": cls.SARVAM_API_KEY,
        }
        
        missing = [key for key, value in required_settings.items() if not value]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set them in your .env file"
            )
        
        return True


# Create a singleton instance
settings = Settings()