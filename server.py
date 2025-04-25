# server.py
import json
import requests
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend access (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more specific in production, e.g., ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:4b" # Or your preferred Ollama model

async def ollama_response_generator(messages: list):
    """
    Generator function to stream responses from Ollama.
    Yields chunks of the response content as they arrive.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages, # Send the whole history
        "stream": True
    }
    try:
        logger.info(f"Sending request to Ollama: {payload}")
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        full_response_for_log = "" # For logging purposes only
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    # Check if the chunk contains message content
                    if message_chunk := chunk.get("message"):
                        content = message_chunk.get("content")
                        if content:
                            # Yield the content part directly
                            yield content
                            full_response_for_log += content # Append for logging

                    # Check if the stream is done (Ollama specific field)
                    if chunk.get("done"):
                        logger.info(f"Ollama stream finished. Full response: {full_response_for_log}")
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON line from Ollama: {line}")
                except Exception as e:
                    logger.error(f"Error processing Ollama stream chunk: {e}")
                    yield f"\n[Error processing chunk: {e}]" # Send error info to client if needed
                    break

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama ({OLLAMA_URL}): {e}")
        yield f"[Error: Could not connect to Ollama. Is it running? Details: {e}]"
    except Exception as e:
        logger.error(f"An unexpected error occurred in ollama_response_generator: {e}")
        yield f"[Error: An unexpected error occurred. Details: {e}]"


@app.post("/chat")
async def chat(request: Request):
    """
    Handles chat requests, takes conversation history,
    and streams back the response from Ollama.
    """
    try:
        body = await request.json()
        # Expecting messages array from frontend [{role: 'user', content: '...'}, {role: 'assistant', ...}]
        messages = body.get("messages")

        if not messages:
            return {"error": "No messages provided"}, 400 # Use proper HTTP errors

        # The generator function handles the streaming
        return StreamingResponse(
            ollama_response_generator(messages),
            media_type="text/plain" # Or text/event-stream if using SSE
        )

    except json.JSONDecodeError:
        logger.error("Received invalid JSON in request body")
        return {"error": "Invalid JSON body"}, 400
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        return {"error": f"Internal server error: {e}"}, 500

# Add a root endpoint for basic check
@app.get("/")
def read_root():
    return {"message": "Chat server is running"}

# It's good practice to run with uvicorn directly in production,
# but this allows `python server.py` for simple testing.
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with uvicorn on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)