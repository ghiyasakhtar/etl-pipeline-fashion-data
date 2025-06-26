from psycopg2 import sql
from psycopg2.sql import Composed
from psycopg2.extras import execute_values
from psycopg2 import Error as Psycopg2Error
from utils.constants import DATA_SHEET_NAME
import logging
import os
import gspread
try:
    from gspread.models import ValueInputOption
except ImportError:
    class ValueInputOption:
        USER_ENTERED = "USER_ENTERED"
from typing import Dict
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread.exceptions import GSpreadException
import psycopg2

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

# Fungsi untuk memformat DataFrame agar sesuai dengan format yang bisa diterima Google Sheets
def _format_data_for_gsheets(df: pd.DataFrame) -> list:
    df_gspread = df.copy()
    # Konversi kolom timestamp ke format string jika bertipe datetime
    if "timestamp" in df_gspread.columns and pd.api.types.is_datetime64_any_dtype(
        df_gspread["timestamp"]
    ):
        df_gspread["timestamp"] = df_gspread["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )

    # Ganti NaN dengan None agar bisa diproses oleh gspread
    df_gspread = df_gspread.astype(object).where(pd.notnull(df_gspread), None)
    # Gabungkan nama kolom dan isinya ke dalam list dua dimensi
    return [df_gspread.columns.values.tolist()] + df_gspread.values.tolist()

# Fungsi untuk menyimpan DataFrame ke file CSV
def save_dataframe_to_csv(df: pd.DataFrame, csv_path: str) -> bool:
    if df.empty:
        logging.warning("Empty DataFrame detected. Not saving to CSV at %s.", csv_path)
        return True

    try:
        # Buat direktori jika belum ada
        dir_name = os.path.dirname(csv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Simpan ke file CSV
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info("Successfully saved %d rows to the CSV file: %s.", len(df), csv_path)
        return True

    # except (OSError, IOError) as e:
    #     logging.error("Failed to data to CSV file %s: %s", csv_path, e)
    #     return False

    except Exception as e:
        if isinstance(e, (OSError, IOError)):
            logging.error("Failed to save data to CSV at %s: %s", csv_path, e)
        else:
            logging.error(
                "An error not anticipated happened while saving the CSV to %s: %s",
                csv_path,
                e,
                exc_info=True,
            )
            logging.error("An unforeseen error occurred in pandas.", exc_info=True)
        return False


# Fungsi untuk membuat skema PostgreSQL berdasarkan DataFrame
def _generate_postgres_schema(df: pd.DataFrame) -> Composed:
    # Pemetaan tipe data pandas ke tipe data PostgreSQL
    type_map = {
        "object": "TEXT",
        "int64": "INTEGER",
        "float64": "REAL",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }

    column_definitions = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        pg_type = None
        # Tangani tipe datetime dengan timezone
        if pd.api.types.is_datetime64_any_dtype(dtype) and getattr(dtype, "tz", None):
            pg_type = "TIMESTAMPTZ"
        elif dtype.name in type_map:
            pg_type = type_map[dtype.name]
            # Atur tipe khusus untuk kolom harga
            if col_name == "price" and pg_type == "REAL":
                pg_type = "NUMERIC(12, 2)"
        else:
            logging.warning(
                "Unknown dtype '%s' in column '%s'. Defaulting to TEXT format.",
                dtype.name,
                col_name,
            )
            pg_type = "TEXT"

        # Tambahkan definisi kolom ke dalam list
        column_definitions.append(
            sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(pg_type))
        )

    # Gabungkan semua definisi kolom menjadi satu pernyataan SQL
    return sql.SQL(", ").join(column_definitions)

# Fungsi untuk mengunggah DataFrame ke Google Sheets
def upload_to_google_sheets(
    df: pd.DataFrame,
    credentials_path: str,
    sheet_id: str,
    worksheet_name: str = DATA_SHEET_NAME,
) -> bool:
    if df.empty:
        logging.warning(
            "The DataFrame is empty, so loading to Google Sheets will be skipped."
        )
        return True

    try:
        # Autentikasi dengan file kredensial
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(sheet_id)

        try:
            # Buka worksheet jika sudah ada
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # Jika worksheet belum ada, buat worksheet baru
            logging.info("Worksheet '%s' was not found. Creating a new one.", worksheet_name)
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1, cols=1)

        # Format data untuk diunggah
        data_to_load = _format_data_for_gsheets(df)

        # Bersihkan worksheet dan unggah data baru
        worksheet.clear()
        worksheet.update(data_to_load, value_input_option=ValueInputOption.USER_ENTERED)  # type: ignore

        logging.info(
            "Successfully loaded %d rows into Google Sheet ID: %s, Worksheet: %s.",
            len(df),
            sheet_id,
            worksheet_name,
        )
        return True

    except GSpreadException as e:
        logging.error("Google Sheets API or client encountered an error: %s.", e)
        return False
    except FileNotFoundError:
        logging.error("Credentials file missing at specified path: %s", credentials_path)
        return False
    except ValueError as e:
        logging.error("A value error occurred during the Google Sheets operation: Invalid data format detected.")
        return False
    except Exception as e:
        logging.error("An unforeseen error happened during Google Sheets loading.")
        logging.error("Underlying error: %s.", e, exc_info=True)
        return False

# Fungsi untuk mengunggah DataFrame ke tabel PostgreSQL
def upload_dataframe_to_postgres(
    df: pd.DataFrame, database_config: Dict[str, str], pg_table_name: str
) -> bool:
    if df.empty:
        logging.warning(
            "The DataFrame is empty, so loading to PostgreSQL table '%s' will be skipped.", pg_table_name
        )
        return True

    conn = None
    try:
        # Koneksi ke database PostgreSQL
        conn = psycopg2.connect(**database_config)
        conn.autocommit = False
        with conn.cursor() as cursor:
            logging.info(
                "Connected to PostgreSQL database '%s'", database_config.get("dbname")
            )

            # Bangun skema tabel berdasarkan DataFrame
            schema_sql = _generate_postgres_schema(df)
            create_table_query = sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ({schema});"
            ).format(table=sql.Identifier(pg_table_name), schema=schema_sql)
            cursor.execute(create_table_query)
            logging.info("Table '%s' has been confirmed to exist.", pg_table_name)

            # Truncate isi tabel sebelum memasukkan data baru
            truncate_query = sql.SQL("TRUNCATE TABLE {table};").format(
                table=sql.Identifier(pg_table_name)
            )
            cursor.execute(truncate_query)
            logging.info("Truncated table '%s'.", pg_table_name)

            # Siapkan data dan konversi NaN menjadi None
            df_prepared = df.astype(object).where(pd.notnull(df), None)
            data_tuples = [tuple(x) for x in df_prepared.to_numpy()]

            if not data_tuples:
                logging.info("No valid data tuples available for insertion into PostgreSQL.")
                conn.commit()
                return True

            # Bangun query INSERT
            cols = sql.SQL(", ").join(map(sql.Identifier, df.columns))
            insert_query = sql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
                table=sql.Identifier(pg_table_name), cols=cols
            )
            execute_values(
                cursor, insert_query.as_string(cursor.connection), data_tuples
            )
            logging.info(
                "Attempting to insert %d records into PostgreSQL table '%s'.",
                len(data_tuples),
                pg_table_name,
            )

        # Commit transaksi
        conn.commit()
        logging.info(
            "Successfully inserted %d records into PostgreSQL table '%s'. Changes have been committed.",
            len(data_tuples),
            pg_table_name,
        )
        logging.info("Load completed successfully. Data inserted into the PostgreSQL table '%s'.", pg_table_name)
        return True

    except Psycopg2Error as e:
        if "Truncate failed" in str(e):
            logging.error("PostgreSQL error while loading table '%s': Failed to truncate table.", pg_table_name)
        elif "Insert failed" in str(e) or "Insert operation failed" in str(e):
            logging.error("PostgreSQL error while loading table '%s': Insert operation failed.", pg_table_name)
        elif "Connection failed" in str(e):
            logging.error("PostgreSQL error while loading table '%s': Connection failure.", pg_table_name)
        else:
            logging.error("PostgreSQL error while loading table '%s': %s.", pg_table_name, e)

        if conn:
            conn.rollback()
            logging.info("PostgreSQL transaction has been rolled back.")
        return False
    except KeyError as e:
        logging.error("Missing required key in db_config: %s.", e.args[0])
        return False
    except Exception as e:
        logging.error(
            "An unexpected error occurred while loading data into PostgreSQL table '%s'.",
            pg_table_name,
        )
        if "commit" in str(e).lower():
            logging.error("Unexpected error during commit operation.")
        else:
            logging.error("An unexpected error occurred during the execute_values operation.")
        if conn:
            conn.rollback()
            logging.info("PostgreSQL transaction has been rolled back.")
        return False
    finally:
        # Tutup koneksi database
        if conn:
            conn.close()
            logging.info("PostgreSQL connection has been closed.")
