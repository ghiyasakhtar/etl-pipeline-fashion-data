PG_TABLE_NAME = "products"
CSV_FILE_LOCATION = "products.csv"

SCHEMA_TYPE_MAP = {
    "title": str,
    "price": float,
    "rating": float,
    "colors": int,
    "size": str,
    "gender": str,
    "image_url": str,
}

DELAY_DURATION = 0.3
USD_IDR_EXCHANGE_RATE = 16000
LOG_MESSAGE_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

EXPECTED_COLUMNS = list(SCHEMA_TYPE_MAP.keys()) + ["timestamp"]
TIMEOUT_DURATION = 15
DATA_SHEET_NAME = "products"
