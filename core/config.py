# core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_TITLE = "Data Analysis Platform"
API_VERSION = "2.0.0"
API_DESCRIPTION = "An intelligent API for automated data analysis, powered by Google Gemini."

# --- Google Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- LangSmith Tracing (Optional) ---
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# --- Model Configuration ---
# Using a Gemini model. "gemini-1.5-flash" is a great, fast choice.
DEFAULT_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

def get_chat_model():
    """
    Get a ChatGoogleGenerativeAI model instance with the current configuration.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        
        return ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            # The 'safety_settings' parameter can be added here if needed to adjust content filtering
        )
    except ImportError:
        raise ImportError("Could not import ChatGoogleGenerativeAI. Please run 'pip install langchain-google-genai'.")
    except Exception as e:
        print(f"Error initializing Gemini chat model: {e}")
        return None