from utils.extract import crawl_all_pages
from utils.load import save_dataframe_to_csv, upload_to_google_sheets, upload_dataframe_to_postgres
from utils.transform import clean_and_transform
import logging
import os
import sys
from utils.constants import (
    PG_TABLE_NAME as DEFAULT_PG_TABLE,
    DATA_SHEET_NAME as DEFAULT_SHEET_NAME,
    CSV_FILE_LOCATION as DEFAULT_CSV_PATH,
)
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import pandas as pd

# Konfigurasi logging untuk mencetak informasi log ke konsol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Fungsi untuk menyimpan DataFrame ke file CSV
def save_to_csv(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    path = cfg["csv_path"]
    if not path.endswith(".csv"):
        logging.warning("CSV output path doesn't end with .csv: %s", path)

    logging.info("Writing data to CSV: %s", path)
    try:
        if save_dataframe_to_csv(df, path):
            logging.info("CSV saved successfully.")
        else:
            logging.error("Failed to save CSV file.")
    except Exception as err:
        logging.error("CSV save error: %s", err, exc_info=True)

# Fungsi untuk memuat konfigurasi dari environment variable
def load_settings() -> Dict[str, Any]:
    load_dotenv()  # Memuat variabel lingkungan dari file .env

    data_url = os.getenv("DATA_SOURCE_URL")
    if not data_url:
        logging.error("DATA_SOURCE_URL is not defined. Aborting pipeline.")
        raise ValueError("Missing data source URL.")

    # Mengambil dan mengonversi SCRAPE_LIMIT, jika tidak valid akan memakai default
    try:
        scrape_limit = int(os.getenv("SCRAPE_LIMIT", "50"))
    except ValueError:
        logging.warning("SCRAPE_LIMIT is not a valid number. Using default: 50")
        scrape_limit = 50

    # Ambil path untuk output CSV
    csv_path = os.getenv("OUTPUT_CSV_PATH", DEFAULT_CSV_PATH)

    # Konfigurasi untuk Google Sheets
    gsheet_creds = os.getenv("GSHEETS_CREDS_PATH")
    gsheet_id = os.getenv("GSHEETS_SHEET_ID")
    gsheets_config = {
        "creds_path": gsheet_creds,
        "sheet_id": gsheet_id,
        "worksheet": os.getenv("WORKSHEET_NAME", DEFAULT_SHEET_NAME),
        "enabled": bool(gsheet_creds and gsheet_id),
    }

    # Konfigurasi untuk PostgreSQL
    db_cfg = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    is_db_ready = all(db_cfg.get(k) for k in ["host", "dbname", "user", "password"])

    postgres_config = {
        "db_cfg": db_cfg,
        "table": os.getenv("POSTGRES_TABLE_NAME", DEFAULT_PG_TABLE),
        "enabled": is_db_ready,
    }

    # Gabungkan semua konfigurasi ke dalam dictionary settings
    settings: Dict[str, Any] = {
        "data_url": data_url,
        "scrape_limit": scrape_limit,
        "csv_path": csv_path,
        "gsheets": gsheets_config,
        "postgres": postgres_config,
    }

    return settings

# Fungsi untuk menyimpan DataFrame ke database PostgreSQL
def save_to_postgres(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    pg = cfg["postgres"]
    if not pg["enabled"]:
        logging.info("PostgreSQL not configured properly. Skipping DB load.")
        return

    db_info = pg["db_cfg"]
    table = pg["table"]

    logging.info("Inserting data into PostgreSQL (%s.%s)...", db_info["dbname"], table)
    try:
        if upload_dataframe_to_postgres(df, db_info, table):
            logging.info("PostgreSQL load complete.")
        else:
            logging.error("Failed to load data into PostgreSQL.")
    except Exception as err:
        logging.error("Postgres load error: %s", err, exc_info=True)

# Fungsi untuk mengekstrak data mentah dari sumber
def extract_raw_data(cfg: Dict[str, Any]) -> Optional[list]:
    logging.info("Step 1: Extracting data from %s (limit: %d pages)...",
                 cfg["data_url"], cfg["scrape_limit"])
    try:
        results = crawl_all_pages(cfg["data_url"], cfg["scrape_limit"])
        if not results:
            logging.warning("No data was scraped.")
            return None
        logging.info("Extracted %d raw items.", len(results))
        return results
    except Exception as err:
        logging.error("Extraction error: %s", err, exc_info=True)
        return None

# Fungsi utama untuk menjalankan seluruh proses ETL
def execute_etl(settings: Dict[str, Any]) -> None:
    logging.info("==============================================")
    logging.info("Starting custom ETL workflow")
    logging.info("==============================================")

    # Langkah ekstraksi
    raw_data = extract_raw_data(settings)
    if not raw_data:
        logging.warning("ETL halted: No data extracted.")
        return

    # Langkah transformasi
    cleaned_data = transform_raw_data(raw_data)
    if cleaned_data is None:
        logging.warning("ETL halted: Transformation failed.")
        return

    # Langkah penyimpanan ke CSV, Google Sheets, dan PostgreSQL
    save_to_csv(cleaned_data, settings)
    save_to_gsheets(cleaned_data, settings)
    save_to_postgres(cleaned_data, settings)

    logging.info("ETL process completed successfully!")

# Fungsi untuk menyimpan data ke Google Sheets
def save_to_gsheets(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    gs = cfg["gsheets"]
    if not gs["enabled"]:
        logging.info("Google Sheets not configured. Skipping this step.")
        return

    if not os.path.exists(gs["creds_path"]):
        logging.warning("Google credentials file missing or not specified: %s", gs["creds_path"])
        return

    logging.info("Uploading to Google Sheets: %s (Worksheet: %s)",
                 gs["sheet_id"], gs["worksheet"])
    try:
        if upload_to_google_sheets(df, gs["creds_path"], gs["sheet_id"], gs["worksheet"]):
            logging.info("Google Sheets updated successfully.")
        else:
            logging.error("Google Sheets upload failed.")
    except Exception as err:
        logging.error("Google Sheets error: %s", err, exc_info=True)

# Fungsi untuk membersihkan dan mentransformasi data mentah menjadi DataFrame
def transform_raw_data(items: list) -> Optional[pd.DataFrame]:
    logging.info("Step 2: Cleaning and transforming data...")
    try:
        df = clean_and_transform(items)
        if df is None or df.empty:
            logging.warning("Transformed data is empty. Skipping load phase.")
            return None
        logging.info("Transformation completed. Rows ready: %d", len(df))
        logging.debug("Preview:\n%s", df.head().to_string())
        return df
    except Exception as err:
        logging.error("Transformation failed: %s", err, exc_info=True)
        return None

# Fungsi entry-point untuk menjalankan pipeline ETL
def start() -> None:
    try:
        cfg = load_settings()  # Muat konfigurasi
        execute_etl(cfg)       # Jalankan proses ETL
    except ValueError as ve:
        logging.error("Configuration issue: %s", ve)
        sys.exit(1)
    except Exception as e:
        logging.critical("Unhandled error: %s", e, exc_info=True)
        sys.exit(1)

# Jalankan fungsi start jika file ini dijalankan langsung
if __name__ == "__main__":
    start()