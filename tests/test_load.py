# import psycopg2
from psycopg2 import Error as Psycopg2Error
from psycopg2 import sql
import pytest
import logging
# import os
from unittest.mock import MagicMock, patch
# from unittest.mock import call
# import gspread
from gspread.exceptions import GSpreadException, WorksheetNotFound
from utils import load
# from utils.constants import DATA_SHEET_NAME
import numpy as np
import pandas as pd

@pytest.fixture
def sample_df():
    # Membuat DataFrame contoh dengan data produk
    data = {
        "title": ["Product A", "Product B"],
        "price": [150000.0, 2500000.50],
        "rating": [4.5, 4.8],
        "colors": [3, 1],
        "size": ["M", "XL"],
        "gender": ["Unisex", "Men"],
        "image_url": ["url_a", "url_b"],
        "timestamp": pd.to_datetime(
            ["2024-01-15 10:00:00+07:00", "2024-01-15 11:00:00+07:00"], utc=True
        ).tz_convert("Asia/Jakarta"),
    }
    # Buat DataFrame dan set tipe data kolom
    df = pd.DataFrame(data).astype(
        {
            "title": str,
            "price": float,
            "rating": float,
            "colors": "Int64",
            "size": str,
            "gender": str,
            "image_url": str,
        }
    )
    return df


@pytest.fixture
def empty_df():
    # DataFrame kosong
    return pd.DataFrame()


@pytest.fixture
def mock_db_config():
    # Konfigurasi database palsu untuk testing
    return {
        "host": "localhost",
        "port": "5432",
        "dbname": "testdb",
        "user": "testuser",
        "password": "password",
    }


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_success(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    # Test simpan CSV dengan folder output
    filepath = tmp_path / "output" / "data.csv"
    result = load.save_dataframe_to_csv(sample_df, str(filepath))
    assert result is True
    mock_makedirs.assert_called_once_with(str(tmp_path / "output"), exist_ok=True)
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_no_subdir(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    # Test simpan CSV tanpa subfolder
    filepath = tmp_path / "data.csv"
    result = load.save_dataframe_to_csv(sample_df, str(filepath))
    assert result is True
    mock_makedirs.assert_called_once_with(str(tmp_path), exist_ok=True)
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_empty_df(mock_to_csv, mock_makedirs, empty_df, tmp_path, caplog):
    # Test simpan DataFrame kosong, tidak menyimpan file
    filepath = tmp_path / "empty.csv"
    caplog.set_level(logging.WARNING)
    result = load.save_dataframe_to_csv(empty_df, str(filepath))
    assert result is True
    mock_makedirs.assert_not_called()
    mock_to_csv.assert_not_called()
    assert f"Empty DataFrame detected. Not saving to CSV at {filepath}." in caplog.text


@pytest.mark.parametrize(
    "exception_class, error_message",
    [
        (IOError, "Insufficient disk space"),
        (OSError, "Permission denied"),
    ],
)
def test_load_to_csv_errors(exception_class, error_message, sample_df, tmp_path, caplog):
    filepath = tmp_path / "error.csv"
    caplog.set_level(logging.ERROR)

    with patch("utils.load.os.makedirs") as mock_makedirs, \
         patch("pandas.DataFrame.to_csv", side_effect=exception_class(error_message)) as mock_to_csv:

        result = load.save_dataframe_to_csv(sample_df, str(filepath))

    # Temp
    # print("caplog.records:", caplog.records)
    # print("caplog.text length:", len(caplog.text))
    # print("=== LOG OUTPUT ===")
    # print(caplog.text)
    # print("==================")

    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()

    assert f"Failed to save data to CSV at {filepath}: {error_message}" in caplog.text


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv", side_effect=Exception("Unexpected pandas error"))
def test_load_to_csv_unexpected_error(
    mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog
):
    # Test saat terjadi error tak terduga pada to_csv
    csv_path = tmp_path / "unexpected.csv"
    caplog.set_level(logging.INFO)
    result = load.save_dataframe_to_csv(sample_df, str(csv_path))

    # Temp
    print("caplog.records:", caplog.records)
    print("caplog.text length:", len(caplog.text))
    print("=== LOG OUTPUT ===")
    print(caplog.text)
    print("==================")

    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert (
        f"An error not anticipated happened while saving the CSV to {csv_path}" in caplog.text
    )
    assert "An unforeseen error occurred in pandas." in caplog.text


def test_format_data_for_gsheets_success(sample_df):
    # Test format data untuk Google Sheets dengan beberapa nilai NaN
    df = sample_df.copy()
    df.loc[0, "rating"] = np.nan
    df.loc[1, "timestamp"] = pd.NaT
    df.loc[0, "colors"] = pd.NA
    result_list = load._format_data_for_gsheets(df)
    assert result_list[0] == list(df.columns)
    assert result_list[1][-1] == "2024-01-15 10:00:00+0700"
    assert result_list[2][-1] is None


def test_format_data_for_gsheets_no_timestamp(sample_df):
    # Test format data ketika kolom timestamp tidak ada
    df_no_ts = sample_df.drop(columns=["timestamp"])
    result_list = load._format_data_for_gsheets(df_no_ts)
    assert list(df_no_ts.columns) == result_list[0]


def test_format_data_for_gsheets_timestamp_not_datetime(sample_df):
    # Test format data dengan nilai timestamp bukan datetime
    df_wrong_ts = sample_df.copy()
    df_wrong_ts["timestamp"] = ["today", "yesterday"]
    result_list = load._format_data_for_gsheets(df_wrong_ts)
    assert result_list[1][-1] == "today"


@patch("utils.load._format_data_for_gsheets")
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_worksheet_exists(
    mock_creds, mock_authorize, mock_prepare, sample_df
):
    # Test upload ke Google Sheets saat worksheet sudah ada
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.return_value = mock_worksheet
    prepared_data = [["col1", "col2"], ["a", 1], ["b", 2]]
    mock_prepare.return_value = prepared_data
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_creds.assert_called_once()
    mock_authorize.assert_called_once()
    mock_gc.open_by_key.assert_called_once_with(sheet_id)
    mock_prepare.assert_called_once_with(sample_df)
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_worksheet.clear.assert_called_once()
    mock_worksheet.update.assert_called_once_with(
        prepared_data, value_input_option="USER_ENTERED"
    )


@patch("utils.load._format_data_for_gsheets")
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_create_worksheet(
    mock_creds, mock_authorize, mock_prepare, sample_df
):
    # Test upload ke Google Sheets saat worksheet belum ada, harus buat baru
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_new_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.side_effect = WorksheetNotFound
    mock_spreadsheet.add_worksheet.return_value = mock_new_worksheet
    prepared_data = [["col1"], ["a"]]
    mock_prepare.return_value = prepared_data
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "NewSheet"
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_prepare.assert_called_once_with(sample_df)
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_called_once_with(
        title=worksheet_name, rows=1, cols=1
    )
    mock_new_worksheet.clear.assert_called_once()
    mock_new_worksheet.update.assert_called_once_with(
        prepared_data, value_input_option="USER_ENTERED"
    )


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_empty_df(mock_creds, mock_authorize, empty_df, caplog):
    # Test upload ke Google Sheets dengan DataFrame kosong (skip upload)
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "EmptySheet"
    caplog.set_level(logging.WARNING)
    result = load.upload_to_google_sheets(empty_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_creds.assert_not_called()
    assert f"The DataFrame is empty, so loading to Google Sheets will be skipped." in caplog.text


@patch("utils.load.gspread.authorize")
@patch(
    "utils.load.Credentials.from_service_account_file",
    side_effect=FileNotFoundError("Creds not found"),
)
def test_load_to_gsheets_credentials_not_found(
    mock_creds, mock_authorize, sample_df, caplog
):
    # Test gagal load Google Sheets saat file kredensial tidak ditemukan
    credentials_path = "nonexistent_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_creds.assert_called_once()
    assert f"Credentials file missing at specified path: {credentials_path}" in caplog.text


@patch("utils.load.gspread.authorize", side_effect=GSpreadException("API Error"))
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_gspread_error(mock_creds, mock_authorize, sample_df, caplog):
    # Test error saat API Google Sheets gagal (misal error jaringan)
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_creds.assert_called_once()
    assert "Google Sheets API or client encountered an error: API Error" in caplog.text


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
@patch("utils.load._format_data_for_gsheets", side_effect=ValueError("Bad data format"))
def test_load_to_gsheets_value_error(
    mock_prepare_data, mock_creds, mock_authorize, sample_df, caplog
):
    # Test error jika data yang disiapkan untuk Google Sheets bermasalah
    mock_gc = MagicMock()
    mock_authorize.return_value = mock_gc
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_prepare_data.assert_called_once()
    assert ("A value error occurred during the Google Sheets operation: Invalid data format detected."
            in caplog.text)


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_unexpected_error(
    mock_creds, mock_authorize, sample_df, caplog
):
    # Test error tak terduga saat membuka sheet Google Sheets
    mock_gc = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.side_effect = Exception("Something else went wrong")
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.upload_to_google_sheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    assert "An unforeseen error happened during Google Sheets loading." in caplog.text


def test_generate_postgres_schema_standard_types():
    # Test konversi tipe data pandas ke skema PostgreSQL untuk tipe standar
    df = pd.DataFrame(
        {
            "col_str": ["a", "b"],
            "col_int": [1, 2],
            "col_float": [1.1, 2.2],
            "col_bool": [True, False],
            "col_dt_naive": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "price": [10.5, 20.0],
        }
    ).astype(
        {
            "col_str": "object",
            "col_int": "int64",
            "col_float": "float64",
            "col_bool": "bool",
            "price": "float64",
            "col_dt_naive": "datetime64[ns]",
        }
    )

    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._generate_postgres_schema(df)

        assert '"col_str" TEXT' in schema_sql
        assert '"col_int" INTEGER' in schema_sql
        assert '"col_float" REAL' in schema_sql
        assert '"col_bool" BOOLEAN' in schema_sql
        assert '"col_dt_naive" TIMESTAMP' in schema_sql
        assert '"price" NUMERIC(12, 2)' in schema_sql


def test_generate_postgres_schema_datetime_aware():
    # Test skema PostgreSQL untuk kolom datetime dengan timezone aware
    df = pd.DataFrame(
        {
            "col_dt_aware": pd.to_datetime(
                ["2024-01-01 10:00:00+07:00"], utc=True
            ).tz_convert("Asia/Jakarta")
        }
    )

    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._generate_postgres_schema(df)
        assert '"col_dt_aware" TIMESTAMPTZ' in schema_sql


def test_generate_postgres_schema_unmapped_type(caplog):
    # Test fallback ke TEXT untuk tipe data pandas yang tidak dikenal
    df = pd.DataFrame({"col_complex": [1 + 2j, 3 + 4j]})
    caplog.set_level(logging.WARNING)

    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._generate_postgres_schema(df)
        assert '"col_complex" TEXT' in schema_sql
        assert (
            "Unknown dtype 'complex128' in column 'col_complex'. Defaulting to TEXT format."
            in caplog.text
        )


def setup_mock_conn_cursor(mock_connect):
    # Setup mock koneksi dan cursor untuk testing database
    mock_conn = MagicMock(name="MockConnection")
    mock_cursor = MagicMock(name="MockCursor")
    mock_connect.return_value = mock_conn
    mock_conn.encoding = "utf-8"
    mock_cursor.connection = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None
    mock_cursor.execute.return_value = None
    mock_conn.rollback = MagicMock(name="rollback")
    mock_conn.commit = MagicMock(name="commit")
    mock_conn.close = MagicMock(name="close")

    return mock_conn, mock_cursor


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
@patch("utils.load.execute_values")
def test_load_to_postgres_success_covers_final_commit_log_return(
    mock_execute_values,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    # Setup mock koneksi dan cursor
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT, col2 INT"
    mock_get_schema.return_value = mock_schema_sql
    table_name = "products_final_check"
    caplog.set_level(logging.INFO)

    # Konversi DataFrame jadi tuple dengan None untuk nilai null agar kompatibel dengan psycopg2
    df_prepared_expected = sample_df.astype(object).where(pd.notnull(sample_df), None)
    expected_data_tuples = [tuple(x) for x in df_prepared_expected.to_numpy()]
    expected_row_count = len(expected_data_tuples)

    # Mock objek SQL dengan string query insert
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = (
        f"INSERT INTO {table_name} (col1, col2) VALUES %s"
    )

    with patch("psycopg2.sql.SQL", return_value=mock_sql_obj), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: x
    ):
        # Panggil fungsi yang dites
        result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Assert hasil sukses dan fungsi return True
    assert (
        result is True
    ), f"The function should return True if successful. Logs: {caplog.text}"

    # Pastikan koneksi DB dibuat dengan benar
    mock_connect.assert_called_once_with(**mock_db_config)
    # Pastikan schema di-generate dari DataFrame
    mock_get_schema.assert_called_once_with(sample_df)

    # Pastikan execute_values dipanggil dengan cursor dan data yang tepat
    mock_execute_values.assert_called_once()
    args = mock_execute_values.call_args[0]
    assert args[0] is mock_cursor  # cursor pertama argumen
    assert len(args) >= 3  # argumen cukup (cursor, sql, data, ...)

    actual_data = args[2]
    assert len(actual_data) == len(
        expected_data_tuples
    ), f"Expected {len(expected_data_tuples)} data tuples, got {len(actual_data)}"

    # Pastikan commit dilakukan sekali, rollback tidak, dan koneksi ditutup
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()
    mock_conn.close.assert_called_once()

    # Log harus memuat pesan sukses
    assert (
        "Load completed successfully." in caplog.text
        and f"into the PostgreSQL table '{table_name}'" in caplog.text
    )


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
@patch("pandas.DataFrame.to_numpy")
@patch("utils.load.execute_values")
def test_load_to_postgres_force_empty_tuples_path(
    mock_execute_values,
    mock_to_numpy,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    assert not sample_df.empty  # pastikan data sample tidak kosong
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"dummy" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "empty_tuples_test"
    caplog.set_level(logging.INFO)
    # Force to_numpy mengembalikan list kosong untuk mensimulasikan kondisi tidak ada data valid
    mock_to_numpy.return_value = []
    result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi tetap sukses dan return True meskipun data tuples kosong
    assert result is True

    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)

    # Pastikan CREATE TABLE dan TRUNCATE TABLE tetap dijalankan
    calls = mock_cursor.execute.call_args_list
    assert any("CREATE TABLE IF NOT EXISTS" in str(c.args[0]) for c in calls)
    assert any("TRUNCATE TABLE" in str(c.args[0]) for c in calls)

    mock_to_numpy.assert_called()
    # Log harus mencatat tidak ada data untuk insert
    assert "No valid data tuples available for insertion into PostgreSQL." in caplog.text

    mock_conn.commit.assert_called_once()
    mock_execute_values.assert_not_called()  # Insert tidak dijalankan
    mock_conn.rollback.assert_not_called()
    mock_conn.close.assert_called_once()


@patch("utils.load.psycopg2.connect")
@patch("utils.load.execute_values")
def test_load_to_postgres_empty_df(
    mock_execute_values, mock_connect, empty_df, mock_db_config, caplog
):
    table_name = "empty_test"
    caplog.set_level(logging.WARNING)

    # Panggil fungsi dengan DataFrame kosong
    result = load.upload_dataframe_to_postgres(empty_df, mock_db_config, table_name)

    # Fungsi skip load, return True, tidak konek ke DB dan tidak insert
    assert result is True
    mock_connect.assert_not_called()
    mock_execute_values.assert_not_called()

    # Log peringatan harus muncul
    assert (
        f"The DataFrame is empty, so loading to PostgreSQL table '{table_name}' will be skipped."
        in caplog.text
    )


@patch("utils.load.psycopg2.connect", side_effect=Psycopg2Error("Connection failed"))
def test_load_to_postgres_connection_error(
    mock_connect, sample_df, mock_db_config, caplog
):
    table_name = "conn_error_test"
    caplog.set_level(logging.ERROR)

    # Panggil fungsi dengan mock koneksi gagal (side effect raise exception)
    result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi return False karena gagal koneksi
    assert result is False
    mock_connect.assert_called_once_with(**mock_db_config)

    # Log error koneksi harus muncul
    assert (
        f"PostgreSQL error while loading table '{table_name}': Connection failure."
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
def test_load_to_postgres_execute_error(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "exec_error_test"
    caplog.set_level(logging.ERROR)

    truncate_error = Psycopg2Error("Truncate failed")

    # Side effect untuk execute: raise error saat menjalankan query TRUNCATE TABLE
    def execute_side_effect(query, *args):
        query_str = str(query)
        if "TRUNCATE TABLE" in query_str:
            raise truncate_error
        return None

    mock_cursor.execute.side_effect = execute_side_effect

    # Panggil fungsi, diharapkan error ditangkap
    result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi return False karena error pada execute
    assert result is False

    mock_get_schema.assert_called_once_with(sample_df)
    assert mock_cursor.execute.call_count >= 2

    # Pastikan rollback dan close dipanggil, commit tidak
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

    # Log error saat eksekusi query muncul
    assert (
        f"PostgreSQL error while loading table '{table_name}': Failed to truncate table."
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
@patch("utils.load.execute_values", side_effect=Psycopg2Error("Insert failed"))
@patch("psycopg2.sql.Composed.as_string")
def test_load_to_postgres_execute_values_error(
    mock_as_string,
    mock_execute_values,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    # Setup mock koneksi dan cursor
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "insert_error_test"
    caplog.set_level(logging.ERROR)
    # Mock string query insert yang dikembalikan oleh as_string
    mock_as_string.return_value = f'INSERT INTO "{table_name}" (...) VALUES %s'

    # Panggil fungsi upload, yang akan gagal saat execute_values karena side effect Psycopg2Error
    result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi harus mengembalikan False karena error saat insert
    assert result is False

    # Pastikan fungsi _generate_postgres_schema dipanggil sekali dengan sample_df
    mock_get_schema.assert_called_once_with(sample_df)
    # Pastikan as_string dipanggil minimal sekali (untuk query insert)
    assert mock_as_string.call_count > 0
    # Pastikan execute_values dipanggil sekali dan menghasilkan error
    mock_execute_values.assert_called_once()

    # Commit tidak boleh dipanggil karena terjadi error
    mock_conn.commit.assert_not_called()
    # Rollback harus dipanggil karena error terjadi
    mock_conn.rollback.assert_called_once()
    # Koneksi harus ditutup
    mock_conn.close.assert_called_once()

    # Log error harus memuat pesan yang sesuai dengan error insert
    assert (
        f"PostgreSQL error while loading table '{table_name}': Insert operation failed."
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
def test_load_to_postgres_missing_db_config_key(
    mock_connect, sample_df, mock_db_config, caplog
):
    # Buat salinan konfigurasi DB dan hapus satu key (misal 'password') untuk simulasi konfigurasi tidak lengkap
    incomplete_config = mock_db_config.copy()
    missing_key = "password"
    del incomplete_config[missing_key]
    table_name = "config_error_test"
    caplog.set_level(logging.ERROR)

    # Mock connect agar raise KeyError karena konfigurasi tidak lengkap
    mock_connect.side_effect = KeyError(missing_key)

    # Panggil fungsi upload dengan konfigurasi yang tidak lengkap, diharapkan gagal
    result = load.upload_dataframe_to_postgres(sample_df, incomplete_config, table_name)

    # Fungsi harus return False karena konfigurasi DB kurang lengkap
    assert result is False

    # Pastikan connect dipanggil dengan konfigurasi yang tidak lengkap tersebut
    mock_connect.assert_called_once_with(**incomplete_config)

    # Log error harus mencantumkan pesan missing key konfigurasi
    assert f"Missing required key in db_config: {missing_key}." in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
def test_load_to_postgres_unexpected_error_during_commit(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    # Setup mock koneksi dan cursor
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT"
    mock_get_schema.return_value = mock_schema_sql
    table_name = "unexpected_commit_err_test"
    caplog.set_level(logging.INFO)

    # Mock objek SQL untuk query insert
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = f"INSERT INTO {table_name} (col1) VALUES %s"

    # Simulasikan error saat commit transaksi
    mock_conn.commit.side_effect = Exception("Unexpected commit error")

    with patch("utils.load.execute_values") as mock_execute_values, patch(
        "psycopg2.sql.SQL", return_value=mock_sql_obj
    ), patch("psycopg2.sql.Identifier", side_effect=lambda x: x):

        # Panggil fungsi upload yang akan error saat commit
        result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi harus return False karena error saat commit
    assert result is False

    # Pastikan fungsi dipanggil sesuai yang diharapkan
    mock_connect.assert_called_once()
    mock_get_schema.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

    # Log harus mencatat error unexpected saat commit dan rollback transaksi
    assert (
        f"An unexpected error occurred while loading data into PostgreSQL table '{table_name}'."
        in caplog.text
    )
    assert "Unexpected error during commit operation." in caplog.text
    assert "PostgreSQL transaction has been rolled back." in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._generate_postgres_schema")
def test_load_to_postgres_unexpected_error_during_execute_values(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    # Setup mock koneksi dan cursor
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT"
    mock_get_schema.return_value = mock_schema_sql
    table_name = "unexpected_generic_err_test"
    caplog.set_level(logging.INFO)

    # Mock objek SQL untuk query insert
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = f"INSERT INTO {table_name} (col1) VALUES %s"

    with patch(
        "utils.load.execute_values",
        side_effect=Exception(
            "An unexpected error occurred during execute_values operation."
        ),
    ) as mock_execute_values, patch(
        "psycopg2.sql.SQL", return_value=mock_sql_obj
    ), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: x
    ):

        # Panggil fungsi upload yang error saat execute_values
        result = load.upload_dataframe_to_postgres(sample_df, mock_db_config, table_name)

    # Fungsi harus return False karena error generic terjadi selama execute_values
    assert result is False

    # Pastikan fungsi dan koneksi dipanggil sesuai yang diharapkan
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.close.assert_called_once()

    # Log harus mencatat error generic dan rollback transaksi
    assert (
        f"An unexpected error occurred while loading data into PostgreSQL table '{table_name}'."
        in caplog.text
    )
    assert "An unexpected error occurred during the execute_values operation." in caplog.text
    assert "PostgreSQL transaction has been rolled back." in caplog.text
