"""
Flask WebSocket Server for Vonage Integration - BINARY AUDIO FIX
==============================================
Handles Vonage webhooks and WebSocket connections for real-time audio
Properly handles BINARY audio frames from Vonage
"""

import os
import json
import uuid
import threading
from flask import Flask, request, jsonify
from flask_sock import Sock

from app.ai_calling.service import (
    call_data,
    ConversationHandler,
    AudioBuffer,
    transcribe_sarvam,
    detect_language,
    generate_ai_response,
    synthesize_sarvam,
)
from config import settings

# ============================================================
# FLASK APP SETUP
# ============================================================

flask_app = Flask(__name__)
sock = Sock(flask_app)

print("[FLASK] WebSocket server initialized")


# ============================================================
# WEBHOOK ENDPOINTS
# ============================================================

@flask_app.route('/webhooks/answer', methods=['GET', 'POST'])
def answer_webhook():
    """
    Handle incoming call - return NCCO with greeting in preferred language
    The preferred_language parameter is passed in the URL query string
    """
    
    # Handle both GET (query params) and POST (JSON body)
    if request.method == 'GET':
        data = request.args.to_dict()
    else:
        data = request.get_json() or {}
    
    print(f"\n[WEBHOOK] Received answer webhook ({request.method}):")
    print(json.dumps(data, indent=2))
    
    if not data:
        print("[WEBHOOK] ⚠️  No data received")
        return jsonify([]), 200
    
    call_uuid = data.get('uuid') or data.get('conversation_uuid')
    from_number = data.get('from')
    to_number = data.get('to')
    
    # Get preferred language from query parameter (default to English)
    preferred_language = data.get('preferred_language', 'en-IN')
    
    # Validate language
    if preferred_language not in settings.LANGUAGE_CONFIG:
        print(f"[WEBHOOK] ⚠️  Invalid language '{preferred_language}', defaulting to en-IN")
        preferred_language = 'en-IN'
    
    print(f"\n{'*'*60}")
    print(f"📞 INCOMING CALL")
    print(f"{'*'*60}")
    print(f"From: {from_number}")
    print(f"To: {to_number}")
    print(f"Call UUID: {call_uuid}")
    print(f"Preferred Language: {preferred_language}")
    print(f"{'*'*60}\n")
    
    # Create conversation handler with preferred language
    handler = ConversationHandler(call_uuid, preferred_language=preferred_language)
    call_data[call_uuid] = handler
    
    # Get greeting in preferred language
    lang_config = settings.LANGUAGE_CONFIG[preferred_language]
    greeting = lang_config["greeting"]
    
    # Log greeting
    handler.add_entry("AI", greeting)
    
    # Construct WebSocket URI properly
    base_url = settings.BASE_URL
    if base_url.startswith('https://'):
        ws_uri = 'wss://' + base_url[8:]
    elif base_url.startswith('http://'):
        ws_uri = 'ws://' + base_url[7:]
    else:
        ws_uri = 'wss://' + base_url
    
    ws_uri = f"{ws_uri}/socket/{call_uuid}"
    
    print(f"[INFO] 🔗 WebSocket URI: {ws_uri}")
    print(f"[INFO] 🗣️  Greeting Language: {lang_config['name']}")
    
    # Generate greeting audio using Sarvam AI
    print(f"[INFO] 🔊 Generating greeting audio...")
    greeting_audio = synthesize_sarvam(greeting, preferred_language)
    
    # NCCO: Connect to WebSocket immediately
    ncco = [
        {
            "action": "connect",
            "eventUrl": [f"{settings.BASE_URL}/webhooks/event"],
            "from": settings.VONAGE_FROM_NUMBER,
            "endpoint": [
                {
                    "type": "websocket",
                    "uri": ws_uri,
                    "content-type": "audio/l16;rate=16000",
                    "headers": {
                        "call_uuid": call_uuid,
                        "app_id": settings.VONAGE_APPLICATION_ID,
                        "preferred_language": preferred_language
                    }
                }
            ]
        }
    ]
    
    # Store greeting audio for WebSocket to send
    if greeting_audio:
        if not hasattr(flask_app, 'greeting_cache'):
            flask_app.greeting_cache = {}
        flask_app.greeting_cache[call_uuid] = greeting_audio
        print(f"[INFO] ✅ Cached greeting audio ({len(greeting_audio)} bytes)")
    else:
        print(f"[ERROR] ❌ Failed to generate greeting audio!")
    
    print(f"[INFO] 📤 Returning NCCO (WebSocket connection)")
    print(json.dumps(ncco, indent=2))
    print()
    
    return jsonify(ncco)


@flask_app.route('/webhooks/event', methods=['GET', 'POST'])
def event_webhook():
    """Handle call events"""
    
    # Handle both GET and POST
    if request.method == 'GET':
        data = request.args.to_dict()
    else:
        data = request.get_json() or {}
    
    if not data:
        return ('', 200)
    
    event_type = data.get('status')
    call_uuid = data.get('uuid') or data.get('conversation_uuid')
    
    print(f"[EVENT] {event_type} | Call: {call_uuid}")
    
    # Save transcript on completion
    if event_type == 'completed' and call_uuid in call_data:
        handler = call_data[call_uuid]
        handler.is_active = False
        filename = handler.save_transcript()
        
        lang_name = settings.LANGUAGE_CONFIG.get(handler.current_language, {}).get("name", handler.current_language)
        print(f"[SUCCESS] ✅ Transcript saved: {filename} (Language: {lang_name})")
        
        # Cleanup
        del call_data[call_uuid]
        
        # Also cleanup greeting cache if exists
        if hasattr(flask_app, 'greeting_cache') and call_uuid in flask_app.greeting_cache:
            del flask_app.greeting_cache[call_uuid]
    
    return ('', 200)


# ============================================================
# WEBSOCKET ENDPOINT - FIXED FOR BINARY AUDIO
# ============================================================

@sock.route('/socket/<call_uuid>')
def websocket_handler(ws, call_uuid):
    """Handle WebSocket connection for real-time audio streaming"""
    print(f"\n[WS] 🔌 WebSocket connected: {call_uuid}")
    
    if call_uuid not in call_data:
        print(f"[WS] ⚠️  Unknown call UUID: {call_uuid}")
        print(f"[WS] Available calls: {list(call_data.keys())}")
        return
    
    handler = call_data[call_uuid]
    audio_buffer = AudioBuffer()
    
    # Track stats
    message_count = 0
    audio_chunks_received = 0
    greeting_sent = False
    
    try:
        # First, send the greeting immediately
        print(f"[WS] 📤 Sending greeting in {handler.current_language}...")
        
        if hasattr(flask_app, 'greeting_cache') and call_uuid in flask_app.greeting_cache:
            greeting_audio = flask_app.greeting_cache[call_uuid]
            print(f"[WS] 🔊 Got cached greeting audio ({len(greeting_audio)} bytes)")
            
            # Send greeting audio directly as binary
            # Vonage expects raw L16 PCM audio
            ws.send(greeting_audio)
            greeting_sent = True
            print(f"[WS] ✅ Greeting sent!")
        else:
            print(f"[WS] ⚠️  No cached greeting found, generating on the fly...")
            greeting_text = settings.LANGUAGE_CONFIG[handler.current_language]["greeting"]
            greeting_audio = synthesize_sarvam(greeting_text, handler.current_language)
            
            if greeting_audio:
                ws.send(greeting_audio)
                greeting_sent = True
                print(f"[WS] ✅ Greeting sent!")
            else:
                print(f"[WS] ❌ Failed to generate greeting!")
        
        # Now listen for incoming audio
        while True:
            message = ws.receive()
            message_count += 1
            
            if message is None:
                print(f"[WS] Connection closed (received None)")
                break
            
            # Vonage sends BINARY audio data, not JSON
            # Check if message is bytes (binary audio)
            if isinstance(message, bytes):
                audio_chunks_received += 1
                
                # This is raw L16 PCM audio from the caller
                # Add to buffer
                if audio_buffer.add_chunk(message):
                    # Buffer is ready - process audio
                    audio_data = audio_buffer.get_audio()
                    
                    print(f"\n[WS] 🎤 Processing buffered audio ({len(audio_data)} bytes)")
                    
                    # Transcribe using Sarvam AI
                    transcript = transcribe_sarvam(audio_data, handler.current_language)
                    
                    if transcript:
                        # Detect language
                        detected_lang = detect_language(transcript)
                        
                        # Update language if changed
                        if detected_lang != handler.current_language:
                            handler.update_language(detected_lang)
                        
                        # Log user input
                        handler.add_entry("User", transcript)
                        
                        # Generate AI response
                        ai_response = generate_ai_response(
                            transcript,
                            handler.current_language,
                            handler.context
                        )
                        
                        # Log AI response
                        handler.add_entry("AI", ai_response)
                        
                        # Convert to speech using Sarvam AI
                        audio_response = synthesize_sarvam(
                            ai_response,
                            handler.current_language
                        )
                        
                        if audio_response:
                            print(f"[WS] 🔊 Sending audio response ({len(audio_response)} bytes)")
                            # Send response audio directly as binary
                            ws.send(audio_response)
                            print(f"[WS] ✅ Response sent!")
                        else:
                            print(f"[WS] ⚠️  No audio response generated")
                    else:
                        print(f"[WS] ⚠️  No transcript from STT")
                
                # Log progress every 100 chunks
                if audio_chunks_received % 100 == 0:
                    print(f"[WS] Status: {audio_chunks_received} audio chunks received")
            
            # Sometimes Vonage sends JSON control messages
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    event_type = data.get('event')
                    
                    if message_count <= 5:
                        print(f"[WS] Control message: {event_type}")
                    
                    if event_type == 'start':
                        print(f"[WS] 🎬 Call started")
                    elif event_type == 'stop':
                        print(f"[WS] 🛑 Call stopped")
                        break
                    
                except json.JSONDecodeError:
                    # Not JSON, just log and skip
                    if message_count <= 5:
                        print(f"[WS] Non-JSON string message: {message[:50]}...")
            else:
                print(f"[WS] ⚠️  Unknown message type: {type(message)}")
    
    except Exception as e:
        print(f"[WS] ❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"[WS] 🔌 WebSocket disconnected: {call_uuid}")
        print(f"[WS] Final stats: {message_count} total messages, {audio_chunks_received} audio chunks")


@flask_app.route('/health', methods=['GET'])
def flask_health():
    """Flask health check"""
    return jsonify({
        "status": "healthy",
        "service": "Flask WebSocket Server",
        "active_calls": len([h for h in call_data.values() if h.is_active]),
        "total_calls": len(call_data),
        "port": 5000,
        "supported_languages": list(settings.LANGUAGE_CONFIG.keys())
    })


# ============================================================
# RUN FLASK SERVER
# ============================================================

def run_flask_server():
    """Run Flask server in a separate thread"""
    print(f"\n{'='*60}")
    print(f"🚀 FLASK WEBSOCKET SERVER")
    print(f"{'='*60}")
    print(f"Server: http://0.0.0.0:5000")
    print(f"Webhooks:")
    print(f"  Answer: {settings.BASE_URL}/webhooks/answer")
    print(f"  Events: {settings.BASE_URL}/webhooks/event")
    print(f"WebSocket: ws://<your-domain>/socket/<uuid>")
    print(f"\n🌐 Supported Languages:")
    for lang_code, config in settings.LANGUAGE_CONFIG.items():
        print(f"  • {config['name']}: {lang_code}")
    print(f"{'='*60}\n")
    
    # Use threaded=True to handle multiple concurrent WebSocket connections
    flask_app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    run_flask_server()