from utils.constants import (
    SCHEMA_TYPE_MAP,
    EXPECTED_COLUMNS,
    USD_IDR_EXCHANGE_RATE,
)
import logging
from typing import Dict, List, Optional
import re
import pandas as pd

# Konfigurasi dasar untuk logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

# Fungsi untuk mengubah string rating menjadi float, jika memungkinkan
def parse_rating(raw_rating: Optional[str]) -> Optional[float]:
    if not isinstance(raw_rating, str):
        logging.debug("Could not parse rating: '%s'.", raw_rating)
        return None
    if (
        raw_rating is None
        or "invalid" in raw_rating.lower()
        or "not rated" in raw_rating.lower()
    ):
        return None
    try:
        match = re.search(r"(\d(\.\d)?)\s*(?:/|\s|$)", raw_rating)
        if match:
            return float(match.group(1))
        return float(raw_rating.strip())
    except (ValueError, TypeError):
        logging.debug("Could not parse rating: '%s'.", raw_rating)
        return None

# Menambahkan kolom timestamp dengan waktu saat ini (timezone Asia/Jakarta)
def _append_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Adding timestamp.")
    try:
        current_utc = pd.Timestamp.now(tz="UTC")
        jakarta_time = current_utc.tz_convert("Asia/Jakarta")
        df["timestamp"] = jakarta_time
    except Exception as tz_error:
        logging.warning("Could not convert timezone to Asia/Jakarta. Using UTC.")
        current_utc = pd.Timestamp.now(tz="UTC")
        df["timestamp"] = current_utc
    return df

# Mengonversi harga USD menjadi IDR menggunakan nilai kurs
def _apply_conversion(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Applying business logic for currency conversion.")
    df["price_idr"] = df["price_usd_value"] * USD_IDR_EXCHANGE_RATE
    return df

# Menghapus entri dengan judul yang tidak valid (misalnya 'Unknown Product')
def _filter_invalid_entries(df: pd.DataFrame) -> pd.DataFrame:
    starting_rows = len(df)
    if df.empty or "title" not in df.columns:
        return df

    invalid_title_mask = df["title"].str.contains("Unknown Product", na=False, case=False)
    df_filtered = df[~invalid_title_mask].copy()
    filtered_row_count = len(df_filtered)
    if starting_rows > filtered_row_count:
        logging.info(
            "Filtered out %d rows with the title 'Unknown Product'.",
            starting_rows - filtered_row_count,
        )
    return df_filtered

# Fungsi untuk mengekstrak jumlah warna dari string
def parse_colors(raw_colors: Optional[str]) -> Optional[int]:
    if raw_colors is None or not isinstance(raw_colors, str):
        logging.debug("Failed to interpret colors from: '%s'.", raw_colors)
        return None

    try:
        match = re.search(r"(\d+)", raw_colors)
        if match:
            return int(match.group(1))

        if "color" in raw_colors.lower():
            logging.debug(f"Detected the word 'color' but no numeric value in: '{raw_colors}'")
        else:
            logging.debug("Failed to interpret colors from: '%s'.", raw_colors)
        return None
    except (ValueError, TypeError):
        logging.debug("Failed to interpret colors from: '%s'.", raw_colors)
        return None

# Membersihkan kolom teks dan melakukan parsing awal
def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Starting initial cleaning and parsing.")
    for col in ["title", "size", "gender", "image_url"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    df["price_usd_value"] = df["price"].apply(parse_price)
    df["rating_value"] = df["rating"].apply(parse_rating)
    df["color_count"] = df["colors"].apply(parse_colors)
    return df

# Parsing harga dari string menjadi float
def parse_price(raw_price: Optional[str]) -> Optional[float]:
    if not isinstance(raw_price, str):
        logging.debug("Could not parse price: '%s'.", raw_price)
        return None
    if raw_price is None or "N/A" in raw_price.lower():
        return None
    try:
        cleaned_price = re.sub(r"[$,]", "", raw_price).strip()
        return float(cleaned_price)
    except (ValueError, TypeError):
        logging.debug("Could not parse price: '%s'.", raw_price)
        return None

# Menyesuaikan nama dan tipe kolom sesuai skema final
def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Preparing the final schema.")
    column_mapping = {
        "price_idr": "price",
        "rating_value": "rating",
        "color_count": "colors",
    }
    expected_columns = [
        "title",
        "price_idr",
        "rating_value",
        "color_count",
        "size",
        "gender",
        "image_url",
        "timestamp",
    ]
    absent_columns = [col for col in expected_columns if col not in df.columns]
    if absent_columns:
        logging.error(
            "The following expected columns are missing before the final selection."
        )
        return pd.DataFrame()

    filtered_columns_df = df[expected_columns].copy()
    df_renamed = filtered_columns_df.rename(columns=column_mapping)

    col = None
    try:
        final_typed_df = df_renamed.copy()
        for col, expected_type in SCHEMA_TYPE_MAP.items():
            if col in final_typed_df.columns:
                if col != "timestamp":
                    if expected_type == pd.Int64Dtype():
                        final_typed_df[col] = pd.Series(
                            final_typed_df[col].values, dtype=pd.Int64Dtype()
                        )
                    else:
                        final_typed_df[col] = final_typed_df[col].astype(expected_type)

        # Pastikan kolom timestamp bertipe datetime
        if (
            "timestamp" in final_typed_df.columns
            and not pd.api.types.is_datetime64_any_dtype(final_typed_df["timestamp"])
        ):
            final_typed_df["timestamp"] = pd.to_datetime(final_typed_df["timestamp"])

        logging.debug("Final data types have been enforced.")
        return final_typed_df

    except (TypeError, ValueError) as e:
        if col is not None:
            logging.error(
                "An error occurred during the final data type conversion for column."
            )
        else:
            logging.error(
                "An error occurred during the final data type conversion: %s.", e
            )
        return df_renamed

# Menghapus baris yang memiliki nilai null atau duplikat
def _drop_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    starting_rows = len(df)
    df.dropna(subset=EXPECTED_COLUMNS, inplace=True)
    rows_after_na = len(df)
    if starting_rows > rows_after_na:
        logging.info(
            "Removed %d rows containing null values in required columns.",
            starting_rows - rows_after_na,
        )

    if df.empty:
        logging.warning("The DataFrame is empty after removing rows with null values.")
        return df

    rows_before_duplicates = len(df)
    df.drop_duplicates(inplace=True, keep="first")
    rows_after_duplicates = len(df)
    if rows_before_duplicates > rows_after_duplicates:
        logging.info(
            "Removed %d duplicate rows.", rows_before_duplicates - rows_after_duplicates
        )

    value_columns = [
        "price_usd_value",
        "price_idr",
        "rating_value",
        "color_count",
    ]
    value_columns_exist = [col for col in value_columns if col in df.columns]

    if value_columns_exist:
        rows_before_value_na = len(df)
        df.dropna(subset=value_columns_exist, inplace=True)
        rows_after_value_na = len(df)
        if rows_before_value_na > rows_after_value_na:
            logging.info(
                "Removed %d rows containing null values in data columns (%s).",
                rows_before_value_na - rows_after_value_na,
                value_columns_exist,
            )

    return df

# Fungsi utama untuk membersihkan dan mentransformasikan data hasil ekstraksi
def clean_and_transform(extracted_data: List[Dict[str, Optional[str]]]) -> pd.DataFrame:
    if not extracted_data:
        logging.warning(
            "Received an empty list for transformation. Returning an empty DataFrame."
        )
        return pd.DataFrame()

    logging.info("Beginning transformation process for %d raw records.", len(extracted_data))

    try:
        df = pd.DataFrame(extracted_data)
        df = _preprocess_data(df)
        df = _append_timestamp(df)
        df = _filter_invalid_entries(df)
        if df.empty:
            logging.warning("The DataFrame is empty after filtering out invalid rows.")
            return df

        df = _apply_conversion(df)

        df_clean_intermediate = _drop_nulls_and_duplicates(df.copy())
        if df_clean_intermediate.empty:
            logging.warning("The DataFrame is empty after removing nulls and duplicates.")
            return df_clean_intermediate

        df_final_schema = _finalize_columns(df_clean_intermediate)

        if df_final_schema.empty and not df_clean_intermediate.empty:
            logging.error(
                "Final schema preparation failed because the DataFrame became empty."
            )
            return df_final_schema

        df_clean = df_final_schema

        final_rows = len(df_clean)
        starting_rows_for_logging = len(extracted_data)
        logging.info(
            "Transformation finished. Resulting rows: %d", final_rows
        )
        if not df_clean.empty:
            logging.debug("Final DataFrame head:\n%s", df_clean.head().to_string())
            logging.debug("Final data types:\n%s", df_clean.dtypes)

            # Temp
            logging.debug("=== FINAL DF HEAD ===\n%s", df_clean.head().to_string())
            logging.debug("=== FINAL DF FULL ===\n%s", df_clean.to_string())
            logging.debug("=== FINAL DF TYPES ===\n%s", df_clean.dtypes)

        return df_clean

    except KeyError as e:
        logging.error(
            f"An expected key is missing during transformation. Please verify the extraction keys."
        )
        return pd.DataFrame()
    except Exception as e:
        logging.error(
            "An unexpected error occurred during data transformation: %s.",
            e,
            exc_info=True,
        )
        return pd.DataFrame()

