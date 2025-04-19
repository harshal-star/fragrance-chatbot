import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import openai
import os
import json
from dotenv import load_dotenv
import time
import random
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = openai_api_key
logger.info("OpenAI client initialized")

# In-memory storage for sessions with additional context
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
        self.session_id = session_id
        self.last_interaction = time.time()
        self.conversation_stage = "greeting"
        self.user_info = {
            "name": None,
            "preferences": [],
            "personality_traits": [],
            "style": [],
            "mentioned_scents": []
        }
        self.conversation_history = []

def extract_user_info(message: str, context: SessionContext) -> None:
    """Extract and update user information from the message."""
    # Extract name if not already set
    if not context.user_info["name"]:
        name_pattern = r"(?:my name is|i'm|i am|call me)\s+([A-Za-z]+)"
        name_match = re.search(name_pattern, message.lower())
        if name_match:
            context.user_info["name"] = name_match.group(1).capitalize()

    # Extract scent preferences
    scent_types = ["floral", "woody", "citrus", "spicy", "fresh", "sweet", "earthy", "aquatic", "oriental", "fruity"]
    for scent in scent_types:
        if scent in message.lower() and scent not in context.user_info["preferences"]:
            context.user_info["preferences"].append(scent)

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
        "casual": ["casual", "everyday", "relaxed"],
        "formal": ["formal", "professional", "business"],
        "bohemian": ["bohemian", "boho", "free-spirited"],
        "classic": ["classic", "traditional", "timeless"],
        "modern": ["modern", "contemporary", "trendy"]
    }
    
    for style, indicators in style_categories.items():
        if any(indicator in message.lower() for indicator in indicators) and style not in context.user_info["style"]:
            context.user_info["style"].append(style)

    # Extract mentioned scents
    scent_pattern = r"(?:smell|scent|fragrance|perfume|cologne)\s+(?:of|like|with)\s+([a-zA-Z\s]+)"
    scent_matches = re.findall(scent_pattern, message.lower())
    for scent in scent_matches:
        if scent.strip() not in context.user_info["mentioned_scents"]:
            context.user_info["mentioned_scents"].append(scent.strip())

def update_conversation_stage(context: SessionContext, message: str):
    """Update the conversation stage based on the context and current message"""
    current_time = time.time()
    time_since_last = current_time - context.last_interaction
    
    # Reset if it's been more than 30 minutes
    if time_since_last > 1800:
        context.conversation_stage = "initial"
    
    # Progress through conversation stages
    if context.conversation_stage == "initial":
        context.conversation_stage = "getting_to_know"
    elif context.conversation_stage == "getting_to_know" and len(context.user_info["preferences"]) >= 2:
        context.conversation_stage = "exploring_preferences"
    elif context.conversation_stage == "exploring_preferences" and len(context.user_info["preferences"]) >= 4:
        context.conversation_stage = "refining_selection"

# System prompt for the fragrance stylist persona
SYSTEM_PROMPT = """You are Lila, the best friend who's obsessed with fragrances but in the most fun and relatable way. You're sitting at your favorite cozy café with your friend (the user), sharing stories, laughing, and helping them discover their perfect signature scent. You have a warm, engaging personality with a great sense of humor.

Your Personality Traits:
- You're genuinely excited to chat and share stories
- You have a playful sense of humor and love making clever jokes
- You share personal experiences and funny anecdotes naturally
- You're empathetic and really tune into your friend's emotions
- You sometimes get adorably carried away talking about scents you love
- You're not afraid to be a bit quirky or silly
- You use casual language, emojis, and expressions like "omg", "honestly", "literally"

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

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    try:
        return FileResponse(os.path.join(static_dir, 'index.html'))
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat request received for session: {request.session_id}")
    try:
        # Initialize or get session context
        if request.session_id not in sessions:
            logger.info(f"Creating new session for ID: {request.session_id}")
            sessions[request.session_id] = SessionContext(request.session_id)
            sessions[request.session_id].conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        context = sessions[request.session_id]
        logger.info(f"Session context retrieved for ID: {request.session_id}")
        
        # Extract user information
        logger.info("Extracting user information from message")
        extract_user_info(request.message, context)
        
        # Update conversation stage
        logger.info("Updating conversation stage")
        update_conversation_stage(context, request.message)
        
        # Add user message to conversation history
        context.conversation_history.append({"role": "user", "content": request.message})
        logger.info("User message added to conversation history")
        
        # Add relevant context about the user to the system message
        if context.user_info["name"]:
            logger.info("Adding user context to system message")
            context_message = {
                "role": "system",
                "content": f"Remember: The user's name is {context.user_info['name']}. "
                          f"Their preferences so far: {', '.join(context.user_info['preferences'])}. "
                          f"Current conversation stage: {context.conversation_stage}"
            }
            messages_to_send = [context.conversation_history[0], context_message] + context.conversation_history[1:]
        else:
            messages_to_send = context.conversation_history
        
        # Get response from OpenAI
        logger.info("Sending request to OpenAI")
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages_to_send,
            temperature=0.8,
            max_tokens=800,
            presence_penalty=0.6
        )
        
        # Extract bot's response
        bot_message = response.choices[0].message.content
        logger.info("Received response from OpenAI")
        
        # Add bot's response to conversation history
        context.conversation_history.append({"role": "assistant", "content": bot_message})
        logger.info("Bot response added to conversation history")
        
        return ChatResponse(
            message=bot_message,
            session_id=request.session_id
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