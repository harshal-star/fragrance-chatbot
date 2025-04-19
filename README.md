# Fragrance Chatbot

A personalized fragrance recommendation chatbot built with FastAPI and OpenAI.

## Project Structure
```
fragrance-chatbot/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
├── .gitignore          # Git ignore rules
└── static/             # Static files
    ├── index.html      # Frontend
    └── images/         # Image assets
```

## Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Deployment
This application is configured for deployment on Render.com. The `render.yaml` file contains all necessary configuration.

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `PYTHON_VERSION`: Python version (3.9.0)

## Features Demonstrated in POC

- Natural conversation with a friendly fragrance stylist persona
- Session-based conversation memory
- Personality and preference-based interaction
- Basic fragrance recommendation through conversation

## Next Steps

After testing this POC, we can:
1. Add persistent storage for user profiles
2. Implement more sophisticated fragrance recommendation logic
3. Add multiple API endpoints for different stages of the conversation
4. Enhance the UI/UX with a web interface 