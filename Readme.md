# AIaaS Finance Platform

AI as a Service platform for Finance Agencies with AI-powered calling capabilities using Vonage, Sarvam AI, and Google Gemini.

## 🚀 Features

### AI Calling Module
- **Multi-language Support**: English, Hindi, and Tamil
- **Automatic Language Detection**: Dynamically switches based on user speech
- **Real-time Transcription**: Powered by Sarvam AI (saarika:v2.5)
- **Natural Voice Synthesis**: Sarvam AI TTS with Indian voices
- **AI Analysis**: Google Gemini for conversation insights
  - Conversation summaries
  - Sentiment analysis (Positive/Neutral/Negative)
  - Intent classification (Paid/Will Pay/Needs Extension/Dispute/No Response)
  - Payment date extraction
- **Bulk Calling**: Trigger multiple calls with different languages
- **Complete Transcripts**: Full conversation history with timestamps

## 📋 Prerequisites

- Python 3.8+
- Vonage API Account
- Sarvam AI API Key
- Google Gemini API Key
- ngrok (for local development with webhooks)

## 🛠️ Installation

1. **Clone the repository**
```bash
cd aiaas_finance_platform
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Vonage Configuration
VONAGE_API_KEY=your_vonage_api_key
VONAGE_API_SECRET=your_vonage_api_secret
VONAGE_APPLICATION_ID=your_application_id
VONAGE_PRIVATE_KEY_PATH=private.key
VONAGE_FROM_NUMBER=12345678901

# Sarvam AI Configuration
SARVAM_API_KEY=your_sarvam_api_key

# Google Gemini AI Configuration
GOOGLE_API_KEY=your_google_api_key

# Server Configuration
BASE_URL=https://your-ngrok-url.ngrok.io
HOST=127.0.0.1
PORT=8000
```

5. **Add Vonage private key**
Place your `private.key` file in the project root directory.

## 🚀 Running the Application

### Start the FastAPI server
```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000`

### API Documentation
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 📡 API Endpoints

### AI Calling Endpoints

#### 1. Trigger Bulk Calls
**POST** `/ai_calling/trigger_calls`

Trigger multiple AI-powered calls to borrowers.

**Request Body:**
```json
{
  "borrowers": [
    {
      "borrower_id": "BRW123456",
      "phone_number": "+911234567890",
      "preferred_language": "hi-IN"
    },
    {
      "borrower_id": "BRW789012",
      "phone_number": "+911987654321",
      "preferred_language": "en-IN"
    }
  ]
}
```

**Response:**
```json
{
  "total_requests": 2,
  "successful_calls": 2,
  "failed_calls": 0,
  "results": [
    {
      "success": true,
      "call_uuid": "abc123...",
      "status": "initiated",
      "to_number": "+911234567890",
      "language": "hi-IN",
      "borrower_id": "BRW123456"
    }
  ]
}
```

#### 2. Make Single Call
**POST** `/ai_calling/make_call`

**Request Body:**
```json
{
  "to_number": "+911234567890",
  "language": "hi-IN",
  "borrower_id": "BRW123456"
}
```

#### 3. Get Transcript
**GET** `/ai_calling/transcript/{call_uuid}`

Returns complete conversation transcript with AI analysis.

#### 4. Get AI Analysis
**GET** `/ai_calling/analysis/{call_uuid}`

Returns only the AI analysis (summary, sentiment, intent).

#### 5. Health Check
**GET** `/ai_calling/health`

Check service status and active calls.

### Supported Languages

- `en-IN`: English (India)
- `hi-IN`: Hindi
- `ta-IN`: Tamil

## 🔧 Configuration

### Language Settings
Language configurations are defined in `config.py`:

```python
LANGUAGE_CONFIG = {
    "en-IN": {
        "name": "English",
        "speaker": "manisha",
        "enable_preprocessing": False,
        "greeting": "Welcome to our customer support..."
    },
    "hi-IN": {
        "name": "Hindi",
        "speaker": "manisha",
        "enable_preprocessing": True,
        "greeting": "ग्राहक सेवा में आपका स्वागत है..."
    },
    "ta-IN": {
        "name": "Tamil",
        "speaker": "manisha",
        "enable_preprocessing": True,
        "greeting": "வாடிக்கையாளர் சேவைக்கு..."
    }
}
```

## 📊 AI Analysis Features

The platform uses Google Gemini to analyze conversations:

1. **Summary**: Concise overview of the conversation
2. **Sentiment**: Positive, Neutral, or Negative
3. **Intent Classification**:
   - **Paid**: Already made payment
   - **Will Pay**: Committed to payment (with date if mentioned)
   - **Needs Extension**: Requesting more time
   - **Dispute**: Disputing the debt
   - **No Response**: Minimal engagement

## 📁 Project Structure

```
aiaas_finance_platform/
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration and settings
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Environment variables template
├── private.key                 # Vonage private key (add your own)
├── app/
│   ├── __init__.py
│   ├── ai_calling/
│   │   ├── __init__.py
│   │   ├── views.py           # API endpoints
│   │   └── service.py         # Core calling logic
│   └── data_ingestion/
│       ├── __init__.py
│       └── views.py           # Data ingestion endpoints
└── README.md
```

## 🧪 Testing

### Test Bulk Calling
```bash
curl -X POST "http://127.0.0.1:8000/ai_calling/trigger_calls" \
  -H "Content-Type: application/json" \
  -d '{
    "borrowers": [
      {
        "borrower_id": "BRW001",
        "phone_number": "+911234567890",
        "preferred_language": "hi-IN"
      }
    ]
  }'
```

### Test Single Call
```bash
curl -X POST "http://127.0.0.1:8000/ai_calling/make_call" \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+911234567890",
    "language": "hi-IN",
    "borrower_id": "BRW001"
  }'
```

### Get Transcript
```bash
curl "http://127.0.0.1:8000/ai_calling/transcript/{call_uuid}"
```

## 🔐 Security Notes

- Never commit `.env` file or `private.key` to version control
- Use environment variables for all sensitive data
- Restrict CORS in production (update `main.py`)
- Implement authentication for production deployment

## 📝 Development Notes

### Adding New Features
1. Add new endpoints in `app/*/views.py`
2. Add business logic in `app/*/service.py`
3. Update configuration in `config.py` if needed
4. Test using Swagger UI at `/docs`

### WebSocket Implementation
For real-time audio streaming with Vonage, you'll need to implement WebSocket endpoints. The current implementation focuses on REST API endpoints for triggering calls and retrieving results.

## 🐛 Troubleshooting

### Common Issues

1. **Vonage client not initialized**
   - Check your Vonage credentials in `.env`
   - Ensure `private.key` file exists in the project root

2. **Gemini API errors**
   - Verify your Google API key
   - Check if you have sufficient quota

3. **Sarvam AI timeout**
   - This is normal for non-English languages (15s timeout)
   - Retry logic is built-in

## 📄 License

This project is proprietary software for finance agency use.

## 🤝 Support

For support and questions, contact the development team.

## 🔄 Version History

- **v1.0.0** - Initial release with AI calling and bulk calling support