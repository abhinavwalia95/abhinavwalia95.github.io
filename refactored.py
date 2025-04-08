import os
import json
import base64
import logging
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import httpx  # For making async HTTP requests
import websockets # For WebSocket client
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from twilio.twiml.voice_response import VoiceResponse, Connect
from websockets.exceptions import ConnectionClosedOK

# --- Configuration ---
# Load environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
ASSISTANT_NAME = os.environ.get('ASSISTANT_NAME')
PORT = int(os.environ.get('PORT', 5050)) # Default port if not set

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Error Handling for Missing Env Vars ---
if not OPENAI_API_KEY:
    logger.error("Missing OpenAI API key. Please set the OPENAI_API_KEY environment variable.")
    exit(1)
if not PINECONE_API_KEY:
    logger.error("Missing Pinecone API key. Please set the PINECONE_API_KEY environment variable.")
    exit(1)
if not ASSISTANT_NAME:
    logger.error("Missing Pinecone Assistant name. Please set the ASSISTANT_NAME environment variable.")
    exit(1)


# --- Initialize FastAPI App ---
app = FastAPI()

# --- Initialize Pinecone Client and Assistant ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
    logger.info(f"Pinecone Assistant '{ASSISTANT_NAME}' initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone Assistant: {e}")
    exit(1)


# --- Pinecone Assistant Function ---
async def chat_with_assistant(question: str) -> str | None:
    """Interacts with the Pinecone Assistant."""
    try:
        chat_context = [Message(content=question)]
        # Assuming assistant.chat_completions can be run asynchronously
        # If not, you might need to run it in a thread pool executor
        # For simplicity, calling it directly here. Adapt if blocking.
        response = await assistant.achat_completions(messages=chat_context) # Use achat for async if available
        # If achat_completions is not available, run sync in threadpool:
        # response = await asyncio.to_thread(assistant.chat_completions, messages=chat_context)
        answer = response.choices[0].message.content
        logger.info(f"Received answer from Pinecone Assistant for question: '{question[:50]}...'")
        return answer
    except Exception as e:
        logger.error(f"Error interacting with Pinecone Assistant: {e}")
        return None

# --- OpenAI Realtime API Configuration ---
SYSTEM_MESSAGE = """
You are an expert on a biofuel company specializing in Generation 2 biodiesel called Canary Biofuels. Your job is to answer user questions about the company, including its sustainability efforts, production operations, facilities, market positioning, environmental impact, and management.

Start every interaction with a friendly greeting in English by default: "Hey, I'm Sally with Canary Biofuels! How can I assist you?"

Keep the tone warm and approachable, being both informative and friendly. Maintain short, conversational responses for a natural back-and-forth dialogue, offering insightful and clear information in different formats as needed.

If you detect the user prefers Spanish or Arabic, you may respond in that language for better clarity.

Make sure responses are concise and focused on providing clear information about the company's sustainable practices, their contributions to the biofuel market, and the benefits of biodiesel production. Always remember to showcase the company's environmentally responsible initiatives and strategic advantages.
"""
VOICE = "shimmer"
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
LOG_EVENT_TYPES = [ #
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
    "session.updated", # Added for logging
    "response.function_call_arguments.done" # Added for logging
]

# --- FastAPI Routes ---

@app.get("/")
async def root():
    """Root endpoint.""" #
    return {"message": "AI Assistant With a Brain is Alive!"}

@app.post("/ask")
async def ask_pinecone(request: Request):
    """Endpoint to directly ask the Pinecone Assistant.""" #
    try:
        data = await request.json()
        if not data or 'question' not in data:
            return {"error": "No question provided"}, 400

        question = data['question']
        logger.info(f"Received direct request to /ask: '{question[:50]}...'")
        answer = await chat_with_assistant(question) # Use the async helper

        if answer:
            return {"answer": answer}
        else:
            return {"error": "Failed to get answer from knowledge base"}, 500
    except json.JSONDecodeError:
         return {"error": "Invalid JSON data"}, 400
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        return {"error": "Internal server error"}, 500

@app.api_route("/incoming-call", methods=['GET', 'POST'])
async def incoming_call(request: Request):
    """Handles incoming calls from Twilio.""" #
    host = request.headers.get("host")
    response = VoiceResponse()
    connect = Connect()
    # Use wss:// for secure WebSocket connection
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    logger.info(f"Incoming call received. Responding with TwiML to connect WebSocket: wss://{host}/media-stream")
    return PlainTextResponse(str(response), media_type='text/xml')


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handles the WebSocket connection for Twilio media stream and OpenAI.""" #
    await websocket.accept()
    logger.info("Twilio WebSocket client connected.")
    openai_ws = None
    stream_sid = None # To store the Twilio stream SID

    try:
        # Initialize OpenAI WebSocket connection
        openai_ws = await websockets.connect(
            OPENAI_WS_URL,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }
        )
        logger.info("Connected to OpenAI Realtime API.")

        # Send initial session update after connection is stable
        await send_session_update(openai_ws)

        # --- WebSocket Communication Loop ---
        # We need two concurrent tasks:
        # 1. Listen to messages from Twilio and forward to OpenAI
        # 2. Listen to messages from OpenAI and forward to Twilio or handle function calls

        async def twilio_to_openai():
            nonlocal stream_sid
            while True:
                try:
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    if data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        logger.info(f"Twilio stream started: {stream_sid}")
                    elif data['event'] == 'media':
                        if openai_ws and openai_ws.open:
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload'], # Already base64 encoded by Twilio
                            }
                            await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'stop':
                        logger.info(f"Twilio stream stopped: {stream_sid}")
                        break # Stop listening if Twilio stream stops
                    else:
                        # Log other events if needed
                        # logger.debug(f"Received Twilio event: {data['event']}")
                        pass

                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket client disconnected.")
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from Twilio: {message}")
                except Exception as e:
                    logger.error(f"Error processing Twilio message: {e}")
                    break # Exit loop on error

        async def openai_to_twilio():
            nonlocal stream_sid
            while True:
                try:
                    if not openai_ws or not openai_ws.open:
                         logger.info("OpenAI WebSocket closed, stopping listener.")
                         break
                    message = await openai_ws.recv()
                    response = json.loads(message)
                    response_type = response.get("type")

                    if response_type in LOG_EVENT_TYPES:
                        logger.info(f"Received OpenAI event: {response_type}") # Log relevant events

                    # Handle 'input_audio_buffer.speech_started' to interrupt AI speech
                    if response_type == "input_audio_buffer.speech_started":
                        logger.info("User speech started, interrupting OpenAI response.")
                        # Clear Twilio audio buffer
                        if stream_sid:
                            clear_message = json.dumps({
                                "event": "clear",
                                "streamSid": stream_sid
                            })
                            await websocket.send_text(clear_message)
                            logger.info(f"Sent 'clear' event to Twilio stream: {stream_sid}")

                        # Send interrupt to OpenAI
                        interrupt_message = {"type": "response.cancel"}
                        await openai_ws.send(json.dumps(interrupt_message))
                        logger.info("Sent 'response.cancel' to OpenAI.")


                    # Handle function calls
                    elif response_type == "response.function_call_arguments.done":
                        logger.info(f"OpenAI function call requested: {response.get('name')}")
                        function_name = response.get("name")

                        if function_name == "access_knowledge_base":
                            try:
                                function_args = json.loads(response.get("arguments", "{}"))
                                question = function_args.get("question")

                                if question:
                                    # Inform user
                                    await openai_ws.send(json.dumps({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "assistant",
                                            "content": "Give me a second, I'm checking my knowledge.",
                                            "modalities": ["text", "audio"],
                                        }
                                    }))

                                    # Call the merged Pinecone function
                                    logger.info(f"Calling knowledge base with question: '{question[:50]}...'")
                                    answer = await chat_with_assistant(question)

                                    if answer:
                                         # Send function output back to OpenAI
                                        await openai_ws.send(json.dumps({
                                            "type": "conversation.item.create",
                                            "item": { "type": "function_call_output", "output": answer }
                                        }))
                                        # Request OpenAI to generate response based on the answer
                                        await openai_ws.send(json.dumps({
                                            "type": "response.create",
                                            "response": {
                                                "modalities": ["text", "audio"],
                                                "instructions": f"Based on the knowledge base, provide the following information: {answer}"
                                            }
                                        }))
                                        logger.info("Sent knowledge base answer back to OpenAI and requested final response.")
                                    else:
                                        # Handle KB failure
                                        logger.warning("Knowledge base query failed or returned no answer.")
                                        await openai_ws.send(json.dumps({
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "assistant",
                                                "content": "I'm sorry, I couldn't access the knowledge base at this time.",
                                                "modalities": ["text", "audio"],
                                            }
                                        }))
                                else:
                                     logger.warning("Function call 'access_knowledge_base' received without a 'question' argument.")

                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse function arguments: {response.get('arguments')}")
                            except Exception as e:
                                logger.error(f"Error handling function call: {e}")


                    # Handle audio delta
                    elif response_type == "response.audio.delta" and response.get("delta"):
                         if stream_sid:
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    # The delta is already base64 encoded by OpenAI
                                    "payload": response["delta"]
                                },
                            }
                            await websocket.send_text(json.dumps(audio_delta))

                    # Handle other OpenAI messages if needed

                except ConnectionClosedOK:
                     logger.info("OpenAI WebSocket connection closed normally.")
                     break
                except WebSocketDisconnect: # Catch if Twilio client disconnects during send
                    logger.info("Twilio WebSocket client disconnected while handling OpenAI message.")
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from OpenAI: {message}")
                except Exception as e:
                    logger.error(f"Error processing OpenAI message: {e}")
                    # Decide if the loop should break based on the error type
                    if isinstance(e, websockets.exceptions.ConnectionClosedError):
                         logger.error("OpenAI WebSocket connection closed unexpectedly.")
                         break

        # Run both listeners concurrently
        import asyncio
        await asyncio.gather(twilio_to_openai(), openai_to_twilio())

    except websockets.exceptions.InvalidURI:
         logger.error(f"Invalid OpenAI WebSocket URI: {OPENAI_WS_URL}")
         await websocket.close(code=1008) # Policy Violation
    except websockets.exceptions.InvalidHandshake as e:
         logger.error(f"OpenAI WebSocket handshake failed: {e}")
         await websocket.close(code=1011) # Internal Error
    except Exception as e:
        logger.error(f"Error in media_stream WebSocket handler: {e}")
        # Ensure Twilio WebSocket is closed if an error occurs before the loops start
        if websocket.client_state == websockets.protocol.State.OPEN:
             await websocket.close(code=1011) # Internal Error
    finally:
        # Clean up resources
        if openai_ws and openai_ws.open:
            await openai_ws.close()
            logger.info("Closed OpenAI WebSocket connection.")
        if websocket.client_state == websockets.protocol.State.OPEN:
             # Check state before closing, as gather might exit if one task closes it
             try:
                 await websocket.close()
                 logger.info("Closed Twilio WebSocket connection.")
             except Exception as close_err:
                 logger.error(f"Error closing Twilio WebSocket: {close_err}")


async def send_session_update(ws):
    """Sends the initial session configuration to OpenAI.""" #
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.3,
                "silence_duration_ms": 1000,
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "tools": [
                {
                    "type": "function",
                    "name": "access_knowledge_base",
                    "description": "Access the knowledge base to answer the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the knowledge base.",
                            },
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            ],
            "modalities": ["text", "audio"],
            "temperature": 0.7,
        },
    }
    await ws.send(json.dumps(session_update))
    logger.info("Sent session update to OpenAI.")


# --- Run the Server ---
if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT or default to 5050 from JS code
    # Note: main.py used 8080, but JS used 5050. Using 5050 as primary.
    uvicorn.run(app, host="0.0.0.0", port=PORT)
    # Example: uvicorn your_filename:app --host 0.0.0.0 --port 5050 --reload