# ===== COMPREHENSIVE CONSTANTS FILE =====
# All constants for the Data Analysis API project

# ===== SYSTEM CONFIGURATION =====

# API Version and basic settings
API_VERSION_CONFIG = "v1"
TIMEOUT = 180  # seconds
MATPLOTLIB_BACKEND = "Agg"  # Non-interactive backend for Docker

# LangChain Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Chain Configuration
CHAIN_TIMEOUT = 120
MAX_RETRIES = 3

# ===== HTTP AND WEB SCRAPING CONSTANTS =====

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

# ===== DATA PROCESSING CONSTANTS =====

COMMON_DATA_CLASSES = [
    "chart", "data", "list", "table", "grid", "content", "results"
]

HTML_PARSER = "html.parser"

# URL patterns for regex extraction
URL_PATTERN = r"https?://[^\s]+"

# Step messages
STEP_0_DATA_FORMAT_DETECTION = "Step 0: LLM-powered data format detection"
STEP_1_TABLE_EXTRACTION = "Step 1: Extract tables from webpage"
STEP_2_DATA_CLEANING = "Step 2: Clean and prepare extracted data"
STEP_3_CHART_TYPE_DETECTION = "Step 3: Detect appropriate chart type using LLM"
STEP_4_VISUALIZATION = "Step 4: Generate visualization"
STEP_5_YoutubeING = "Step 5: Answer questions about the data"

# English stopwords for keyword extraction
ENGLISH_STOPWORDS = {
    "the", "of", "and", "to", "in", "for", "by", "with", "on", "at", "from",
    "as", "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "a", "an", "or", "that", "this", "which", "who", "what", "when", "where",
    "how", "why", "it", "its", "but", "not",
}

# Content selectors for non-table data extraction
CONTENT_SELECTORS = [
    ".chart", ".data-table", ".list", ".grid", ".content",
    "[class*='chart']", "[class*='table']", "[class*='list']",
]
WORD_REGEX_PATTERN = r"\b\w+\b"
MIN_KEYWORD_LENGTH = 2