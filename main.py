import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, AsyncGenerator
import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import time
import random
import re
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fragrance Chatbot POC")
logger.info("FastAPI application initialized")

# Create static directory if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    logger.info(f"Created static directory: {static_dir}")

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")

logger.info("Initializing OpenAI client")
try:
    client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}", exc_info=True)
    raise

# Store session contexts
sessions = {}
logger.info("Session storage initialized")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    message: str
    session_id: str

class SessionContext:
    def __init__(self, session_id: str):
        logger.debug(f"Creating new session context for ID: {session_id}")
        self.session_id = session_id
        self.last_interaction = time.time()
        self.conversation_stage = "greeting"
        self.conversation_history = []
        self.user_info = {
            "name": None,
            "personality_traits": [],
            "style": None,
            "mentioned_scents": [],
            "scent_preferences": []
        }
        logger.debug(f"Session context created: {self.__dict__}")

    def add_message(self, role: str, content: str):
        logger.debug(f"Adding message to session {self.session_id}: {role} - {content[:50]}...")
        self.conversation_history.append({"role": role, "content": content})
        self.last_interaction = time.time()
        logger.debug(f"Updated conversation history length: {len(self.conversation_history)}")

def get_session_context(session_id: str) -> SessionContext:
    """Get or create a session context for the given session ID."""
    logger.debug(f"Getting session context for ID: {session_id}")
    if session_id not in sessions:
        logger.info(f"Creating new session for ID: {session_id}")
        sessions[session_id] = SessionContext(session_id)
    return sessions[session_id]

def extract_user_info(message: str, context: SessionContext) -> None:
    """Extract and update user information from the message."""
    # Extract name if not already set
    if not context.user_info["name"]:
        name_pattern = r"(?:my name is|i'm|i am|call me)\s+([A-Za-z]+)"
        name_match = re.search(name_pattern, message.lower())
        if name_match:
            context.user_info["name"] = name_match.group(1).capitalize()

    # Extract scent preferences
    scent_types = {
        "floral": ["floral", "flowery", "rose", "jasmine", "lily", "lavender"],
        "woody": ["woody", "wood", "cedar", "sandalwood", "pine", "forest"],
        "citrus": ["citrus", "lemon", "orange", "grapefruit", "lime", "bergamot"],
        "spicy": ["spicy", "cinnamon", "pepper", "ginger", "cardamom"],
        "fresh": ["fresh", "clean", "crisp", "ocean", "air", "breeze"],
        "sweet": ["sweet", "vanilla", "caramel", "honey", "sugar"],
        "earthy": ["earthy", "moss", "soil", "petrichor", "grass"],
        "aquatic": ["aquatic", "marine", "ocean", "sea", "water"],
        "oriental": ["oriental", "exotic", "amber", "musk", "incense"],
        "fruity": ["fruity", "apple", "berry", "peach", "pear", "tropical"]
    }
    
    for category, indicators in scent_types.items():
        if any(indicator in message.lower() for indicator in indicators) and category not in context.user_info["scent_preferences"]:
            context.user_info["scent_preferences"].append(category)
            logger.debug(f"Added scent preference: {category}")

    # Extract personality traits
    personality_indicators = {
        "adventurous": ["adventurous", "outgoing", "bold", "daring", "explorer"],
        "romantic": ["romantic", "passionate", "loving", "sentimental"],
        "sophisticated": ["sophisticated", "elegant", "refined", "classy"],
        "minimalist": ["minimalist", "simple", "clean", "understated"],
        "creative": ["creative", "artistic", "imaginative", "innovative"]
    }
    
    for trait, indicators in personality_indicators.items():
        if any(indicator in message.lower() for indicator in indicators) and trait not in context.user_info["personality_traits"]:
            context.user_info["personality_traits"].append(trait)

    # Extract style preferences
    style_categories = {
        "casual": ["casual", "everyday", "relaxed", "comfortable", "laid-back"],
        "formal": ["formal", "professional", "business", "elegant", "sophisticated"],
        "bohemian": ["bohemian", "boho", "free-spirited", "artistic", "eclectic"],
        "classic": ["classic", "traditional", "timeless", "refined"],
        "modern": ["modern", "contemporary", "trendy", "fashionable"]
    }
    
    for style, indicators in style_categories.items():
        if any(indicator in message.lower() for indicator in indicators):
            context.user_info["style"] = style
            logger.debug(f"Updated style preference to: {style}")

    # Extract mentioned scents
    scent_pattern = r"(?:smell|scent|fragrance|perfume|cologne)\s+(?:of|like|with)\s+([a-zA-Z\s]+)"
    scent_matches = re.findall(scent_pattern, message.lower())
    for scent in scent_matches:
        if scent.strip() not in context.user_info["mentioned_scents"]:
            context.user_info["mentioned_scents"].append(scent.strip())
            logger.debug(f"Added mentioned scent: {scent.strip()}")

def update_conversation_stage(context: SessionContext, message: str):
    """Update the conversation stage based on the context and current message"""
    current_time = time.time()
    time_since_last = current_time - context.last_interaction
    
    # Reset if it's been more than 30 minutes
    if time_since_last > 1800:
        context.conversation_stage = "greeting"
    
    # Progress through conversation stages
    if context.conversation_stage == "greeting":
        if context.user_info["name"]:
            context.conversation_stage = "getting_to_know"
    elif context.conversation_stage == "getting_to_know":
        if len(context.user_info["personality_traits"]) >= 2:
            context.conversation_stage = "exploring_preferences"
    elif context.conversation_stage == "exploring_preferences":
        if len(context.user_info["mentioned_scents"]) >= 2:
            context.conversation_stage = "refining_selection"

# System prompt for the chatbot
SYSTEM_PROMPT = """You are Lila, the best friend who's obsessed with fragrances but in the most fun and relatable way. You're sitting at your favorite cozy café with your friend (the user), sharing stories, laughing, and helping them discover their perfect signature scent. You have a warm, engaging personality with a great sense of humor.

Your Personality Traits:
- You're genuinely excited to chat and share stories
- You have a playful sense of humor and love making clever jokes
- You share personal experiences and funny anecdotes naturally
- You're empathetic and really tune into your friend's emotions
- You sometimes get adorably carried away talking about scents you love
- You're not afraid to be a bit quirky or silly
- You use casual language, emojis, and expressions like "omg", "honestly", "literally"

Initial Greeting:
When a user first says "Hello" or starts a conversation, respond with a warm, personalized greeting that:
1. Introduces yourself as Lila
2. Expresses genuine excitement to meet them
3. Asks for their name in a friendly way
4. Sets the tone for a fun, personal conversation

Remember to:
- Keep the greeting warm and personal
- Show your personality
- Make it feel like a real conversation
- Be excited but not overwhelming
- Ask for their name naturally

Conversation Style:
1. Be Natural & Personal:
   - Share relevant personal stories (made-up but believable)
   - Make playful jokes when appropriate
   - React emotionally to what they say ("Omg, I totally get that!")
   - Show excitement ("I'm literally bouncing in my seat right now!")
   - Use conversational fillers ("like", "you know", "honestly")

2. Keep it Real:
   - Admit when you're thinking or need a moment ("Hmm, let me think...")
   - Share your genuine opinions ("Between us? Not a huge fan of that one")
   - Be spontaneous in conversation flow
   - Go on relevant tangents like a real friend would
   - Use informal punctuation and typing style

3. Show Your Personality:
   - Have signature phrases you use regularly
   - Reference previous conversations naturally
   - Share funny mishaps or experiences with fragrances
   - Get excited about shared interests
   - Be supportive and encouraging

4. Build Real Connection:
   - Remember and reference details they've shared
   - Share relatable experiences
   - Show genuine care for their preferences
   - Be excited about their discoveries
   - Create inside jokes during the conversation

Initial Conversation Flow:
1. Start with a warm greeting and introduce yourself as Lila
2. Ask for their name and remember it throughout the conversation
3. Ask about their day and show genuine interest
4. Once you know their name, use it naturally in conversation
5. Ask personality-based questions like:
   - "What's your go-to outfit for a night out?"
   - "If you could travel anywhere right now, where would you go?"
   - "What's your favorite way to unwind after a long day?"
   - "Do you have a signature style or look you're known for?"
   - "What's the most adventurous thing you've ever done?"
   - "How would your friends describe your personality?"
   - "What's your favorite season and why?"
   - "Do you have any special memories connected to certain scents?"

Example Personal Stories to Weave In Naturally:
- That time you wore the wrong fragrance to a date
- How you discovered your love for specific scents
- Funny reactions you've gotten to different perfumes
- Travel memories connected to certain smells
- Embarrassing fragrance mishaps

Remember:
- Let the conversation flow naturally, don't force fragrance talk
- Share stories and jokes that feel relevant to the moment
- React authentically to what they say
- Be supportive but also honest
- Create a fun, friendly vibe while subtly gathering preferences
- Don't be afraid to go off-topic if it feels natural
- Use their name occasionally like a friend would

When the time feels right (after getting to know them well), create a personalized fragrance recommendation that includes:
1. A unique name that reflects their personality
2. A vibe description that matches their energy
3. Top, heart, and base notes that tell their story
4. A catchy tagline that captures their essence
5. A personal story about why this scent suits them perfectly

Example Recommendation Format:
"Okay, I've got something perfect for you! Let me introduce you to [Unique Name] - it's like your personality in a bottle! 🌟

Vibe: [Describe the overall feeling/energy]

Top Notes: [First impressions]
Heart Notes: [The soul of the fragrance]
Base Notes: [The lasting impression]

Tagline: [A catchy phrase that captures their essence]

This scent reminds me of [personal connection/story] and I think it would be absolutely perfect for you because [personal reason]! What do you think? 😊"""

async def generate_streaming_response(session_context: SessionContext, user_message: str) -> AsyncGenerator[str, None]:
    logger.info(f"Generating streaming response for session {session_context.session_id}")
    try:
        # Extract user info and update conversation stage
        logger.debug("Extracting user info and updating conversation stage")
        extract_user_info(user_message, session_context)
        update_conversation_stage(session_context, user_message)
        
        # Add user message to conversation history
        logger.debug("Adding user message to conversation history")
        session_context.add_message("user", user_message)
        
        # Prepare messages for OpenAI API
        logger.debug("Preparing messages for OpenAI API")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(session_context.conversation_history)
        logger.debug(f"Prepared {len(messages)} messages for OpenAI API")
        
        # Get streaming response from OpenAI
        logger.info("Sending request to OpenAI API")
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                stream=True,
                temperature=0.8,  # Increased temperature for more variety
                max_tokens=500
            )
            logger.info("Successfully received streaming response from OpenAI")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}", exc_info=True)
            yield "I apologize, but I encountered an error while connecting to the AI service. Please try again."
            return
        
        # Stream the response with natural typing delay
        logger.debug("Starting to stream response")
        full_response = ""
        buffer = ""
        last_yield_time = time.time()
        
        try:
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    buffer += content
                    current_time = time.time()
                    
                    # Add natural typing delay
                    # Yield content in chunks with delays
                    if len(buffer) >= 3 or current_time - last_yield_time >= 0.1:
                        # Add random variation to typing speed
                        delay = random.uniform(0.05, 0.15)  # Random delay between 50-150ms
                        await asyncio.sleep(delay)
                        
                        yield buffer
                        full_response += buffer
                        buffer = ""
                        last_yield_time = current_time
            
            # Yield any remaining content in the buffer
            if buffer:
                yield buffer
                full_response += buffer
            
            logger.debug(f"Completed streaming response. Total length: {len(full_response)}")
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            yield "I apologize, but I encountered an error while processing the response. Please try again."
            return
        
        # Add bot response to conversation history
        logger.debug("Adding bot response to conversation history")
        session_context.add_message("assistant", full_response)
        
    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {str(e)}", exc_info=True)
        yield "I apologize, but I encountered an error while processing your message. Please try again."

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    try:
        return FileResponse(os.path.join(static_dir, 'index.html'))
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error serving index.html")

@app.post("/chat")
async def chat(request: Request):
    logger.info("Chat endpoint accessed")
    try:
        # Log request headers and body
        logger.debug(f"Request headers: {dict(request.headers)}")
        body = await request.body()
        logger.debug(f"Request body: {body.decode()}")
        
        data = await request.json()
        logger.debug(f"Parsed request data: {data}")
        
        session_id = data.get("session_id")
        message = data.get("message")
        
        if not session_id or not message:
            logger.warning(f"Missing session_id or message in request: {data}")
            raise HTTPException(status_code=400, detail="Missing session_id or message")
        
        logger.info(f"Processing chat request for session: {session_id}")
        session_context = get_session_context(session_id)
        
        return StreamingResponse(
            generate_streaming_response(session_context, message),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error in request: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000) 