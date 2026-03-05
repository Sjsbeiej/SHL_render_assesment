import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "shl_assessments.json")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# SHL Catalogue
SHL_CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
SHL_PRODUCT_BASE = "https://www.shl.com/solutions/products/product-catalog/view/"

# Scraper settings
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

# Recommendation settings
TOP_K_RETRIEVAL = 40   # candidates from vector search
TOP_K_FINAL = 10       # final recommendations returned
