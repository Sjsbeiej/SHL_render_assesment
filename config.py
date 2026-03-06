import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider Selection: "gpt" or "gemini"
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gpt").lower()  # "gpt" or "gemini"

# OpenAI Configuration (used when LLM_PROVIDER = "gpt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"

# Gemini Configuration (used when LLM_PROVIDER = "gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-3.0-flash"  # best free-tier Gemini model

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "shl_assessments.json")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# SHL Catalogue
SHL_CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
SHL_PRODUCT_BASE = "https://www.shl.com/solutions/products/product-catalog/view/"

# Scraper settings
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

# Recommendation settings
TOP_K_PER_QUERY = 60   # candidates per search query from vector search
TOP_K_TO_LLM = 70      # total candidates sent to LLM reranker
TOP_K_FINAL = 10       # final recommendations to show
