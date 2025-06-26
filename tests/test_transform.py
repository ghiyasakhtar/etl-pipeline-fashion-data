from utils.constants import (
    SCHEMA_TYPE_MAP,
    USD_IDR_EXCHANGE_RATE,
)
# from utils.constants import EXPECTED_COLUMNS

import logging
# from datetime import timezone
from unittest.mock import MagicMock, patch

# noinspection PyProtectedMember
from utils.transform import (
    _append_timestamp,
    _apply_conversion,
    _filter_invalid_entries,
    _preprocess_data,
    _finalize_columns,
    _drop_nulls_and_duplicates,
    parse_colors,
    parse_price,
    parse_rating,
    clean_and_transform,
)

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import mark

SAMPLE_RAW_DATA = [
    dict(
        title="Hoodie Classic",
        price="$25.00",
        rating="⭐ 4.7 / 5",
        colors="4 colors",
        size="XL",
        gender="Men",
        image_url="url1"
    ),
    dict(
        title="Running Shorts",
        price="$18.99",
        rating="⭐ 4.2 / 5",
        colors="3 colors",
        size="M",
        gender="Unisex",
        image_url="url2"
    ),
    dict(
        title="Hoodie Classic",
        price="$25.00",
        rating="⭐ 4.7 / 5",
        colors="4 colors",
        size="XL",
        gender="Men",
        image_url="url1"
    ),
    dict(
        title="Denim Jacket",
        price="N/A",
        rating="⭐ 4.8 / 5",
        colors="2 colors",
        size="L",
        gender="Women",
        image_url="url3"
    ),
    dict(
        title="Unknown Product 1",
        price="$55.99",
        rating="⭐ 4.5 / 5",
        colors="5 Color",
        size="L",
        gender="Men",
        image_url="url4"
    ),
    dict(
        title="Sports Bra",
        price="$22.00",
        rating="N/A",
        colors="3 colors",
        size="S",
        gender="Women",
        image_url="url5"
    ),
    dict(
        title="Casual Pants",
        price="$30.00",
        rating="⭐ 4.3 / 5",
        colors=None,
        size="M",
        gender="Men",
        image_url="url6"
    ),
    dict(
        title="Baseball Cap",
        price="$14.00",
        rating="⭐ 4.1 / 5",
        colors="6 color",
        size=None,
        gender="Unisex",
        image_url="url7"
    ),
    dict(
        title="Wool Scarf",
        price="$19.50",
        rating="⭐ 4.4 / 5",
        colors="3 colors",
        size="XL",
        gender="Women",
        image_url="url8"
    ),
    dict(
        title="Flip Flops",
        price="$10.99",
        rating="⭐ 3.9 / 5",
        colors="2 color",
        size="X",
        gender="Unisex",
        image_url=None
    ),
    dict(
        title="Leather Belt",
        price="Ten Dollars",
        rating="4.3",
        colors="1 color",
        size="M",
        gender="Unisex",
        image_url="url9"
    ),
    dict(
        title="Beanie Hat",
        price="$9.99",
        rating="Trash",
        colors="3 colors",
        size="L",
        gender="Unisex",
        image_url="url10"
    ),
    dict(
        title="Training Gloves",
        price="$14.50",
        rating="3.9",
        colors="Various Color",
        size="L",
        gender="Unisex",
        image_url="url11"
    ),
    dict(
        title="Unknown Product 2",
        price="$28.00",
        rating="4.7",
        colors="5 color",
        size="L",
        gender="Old",
        image_url="url12"
    ),
]

EXPECTED_COLS = []
for column in (
    "title",
    "price",
    "rating",
    "colors",
    "size",
    "gender",
    "image_url",
    "timestamp",
):
    EXPECTED_COLS.append(column)

@mark.parametrize(
    argnames=["price_str", "expected"],
    argvalues=[
        ("$10.00", 10.0),
        (" $ 25.50 ", 25.5),
        ("$1,200.99", 1200.99),
        ("99.9", 99.9),
        ("N/A", None),
        ("unavailable", None),
        (None, None),
        ("", None),
        ("abc", None),
        (100, None),
    ],
)


def test_parse_price(price_str, expected, caplog):
    with caplog.at_level(logging.DEBUG):
        assert parse_price(price_str) == expected
        if expected is None and price_str not in [
            None,
            "Price Unavailable",
            "unavailable",
            "",
        ]:
            assert f"Could not parse price: '{price_str}'" in caplog.text


@pytest.mark.parametrize(
    "rating_input, expected_output",
    [
        ("⭐ 4.5 / 5", 4.5),
        (" 3.8 / 5 stars ", 3.8),
        ("Rating: 4.0", 4.0),
        ("5/5", 5.0),
        ("4", 4.0),
        (" 4.2 ", 4.2),
        ("Not Rated", None),
        ("invalid rating", None),
        (None, None),
        ("", None),
        ("abc", None),
        ("N/A", None),
        ("4/ stars", 4.0),
        (4.5, None),
    ],
)
def test_parse_rating_variation(rating_input, expected_output, caplog):
    with caplog.at_level(logging.DEBUG):
        assert parse_rating(rating_input) == expected_output
        unparsed_values = [None, "Not Rated", "invalid rating", ""]
        if expected_output is None and rating_input not in unparsed_values:
            assert f"Could not parse rating: '{rating_input}'" in caplog.text


@pytest.mark.parametrize(
    "colors_input, expected_result",
    [
        ("3 colors", 3),
        ("Available in 5 Colors", 5),
        (" 1 Color ", 1),
        ("10", 10),
        (None, None),
        ("", None),
        ("Various Color", None),
        ("abc", None),
        (3, None),
    ],
)


def test_parse_colors_variant(colors_input, expected_result, caplog):
    with caplog.at_level(logging.DEBUG):
        assert parse_colors(colors_input) == expected_result
        if expected_result is None and colors_input not in [None, ""]:
            if isinstance(colors_input, str):
                has_color_word = "color" in colors_input.lower()
                has_digit = any(char.isdigit() for char in colors_input)
                if has_color_word and not has_digit:
                    assert f"Detected the word 'color' but no numeric value in: '{colors_input}'" in caplog.text
                elif not (has_color_word and has_digit):
                    assert f"Failed to interpret colors from: '{colors_input}'" in caplog.text
            else:
                assert f"Failed to interpret colors from: '{colors_input}'" in caplog.text


@patch("utils.transform.re.search")
def test_parse_colors_raises_value_error(mock_search, caplog):
    mock_search.side_effect = ValueError("Forced ValueError from regex")
    test_input = "The string provided is valid."
    with caplog.at_level(logging.DEBUG):
        output = parse_colors(test_input)
    assert output is None
    assert f"Failed to interpret colors from: '{test_input}'." in caplog.text
    mock_search.assert_called_once_with(r"(\d+)", test_input)


@patch("utils.transform.re.search")
def test_parse_colors_raises_type_error(mock_search, caplog):
    mock_search.side_effect = TypeError("Forced TypeError from regex")
    test_input = "This is another valid string input."
    with caplog.at_level(logging.DEBUG):
        output = parse_colors(test_input)
    assert output is None
    assert f"Failed to interpret colors from: '{test_input}'." in caplog.text
    mock_search.assert_called_once_with(r"(\d+)", test_input)


def test_parse_colors_with_non_string_input(caplog):
    input_value = 12345
    with caplog.at_level(logging.DEBUG):
        output = parse_colors(input_value)  # type: ignore[arg-type]
    assert output is None
    assert f"Failed to interpret colors from: '{input_value}'." in caplog.text


def test_data_preprocessing():
    raw_df = pd.DataFrame(SAMPLE_RAW_DATA[:2])
    cleaned_df = _preprocess_data(raw_df.copy())

    assert "price_usd_value" in cleaned_df.columns
    assert "rating_value" in cleaned_df.columns
    assert "color_count" in cleaned_df.columns
    assert cleaned_df.loc[0, "title"] == "Hoodie Classic"
    assert cleaned_df.loc[0, "size"] == "XL"
    assert cleaned_df.loc[0, "gender"] == "Men"
    assert cleaned_df.loc[0, "price_usd_value"] == 25.0
    assert cleaned_df.loc[1, "price_usd_value"] == 18.99
    assert cleaned_df.loc[0, "rating_value"] == 4.7
    assert cleaned_df.loc[1, "rating_value"] == 4.2
    assert cleaned_df.loc[0, "color_count"] == 4
    assert cleaned_df.loc[1, "color_count"] == 3


def test_data_preprocessing_with_missing_columns():
    input_data = [{"price": "$10", "rating": "4", "colors": "2"}]
    raw_df = pd.DataFrame(input_data)
    cleaned_df = _preprocess_data(raw_df.copy())
    assert cleaned_df.loc[0, "price_usd_value"] == 10.0
    assert cleaned_df.loc[0, "rating_value"] == 4.0
    assert cleaned_df.loc[0, "color_count"] == 2


@pytest.fixture
def patched_timestamp():
    with patch("pandas.Timestamp") as mock_timestamp:
        yield mock_timestamp


@patch("utils.transform.pd.Timestamp.now")
def test_append_timestamp_with_mock(mock_now):
    jakarta_time = pd.Timestamp("2023-10-27 17:00:00", tz="Asia/Jakarta")

    mock_timestamp = MagicMock(spec=pd.Timestamp)
    mock_timestamp.tz_convert.return_value = jakarta_time
    mock_now.return_value = mock_timestamp

    df_original = pd.DataFrame({"col1": [1, 2]})
    df_with_ts = _append_timestamp(df_original.copy())

    assert "timestamp" in df_with_ts.columns
    expected_df = df_original.copy()
    expected_df["timestamp"] = jakarta_time

    pd.testing.assert_series_equal(df_with_ts["timestamp"], expected_df["timestamp"])
    mock_now.assert_called_once_with(tz="UTC")
    mock_timestamp.tz_convert.assert_called_once_with("Asia/Jakarta")


@patch("utils.transform.pd.Timestamp.now")
def test_append_timestamp_handles_tz_error(mock_now, caplog):
    utc_time = pd.Timestamp("2023-10-27 10:00:00", tz="UTC")
    mock_timestamp_error = MagicMock(spec=pd.Timestamp)
    mock_timestamp_error.tz_convert.side_effect = Exception("TZ database error")

    mock_now.side_effect = [mock_timestamp_error, utc_time]

    df_original = pd.DataFrame({"col1": [1, 2]})
    with caplog.at_level(logging.WARNING):
        df_with_ts = _append_timestamp(df_original.copy())

    assert "timestamp" in df_with_ts.columns
    expected_df = df_original.copy()
    expected_df["timestamp"] = utc_time

    pd.testing.assert_series_equal(df_with_ts["timestamp"], expected_df["timestamp"])
    assert "Could not convert timezone to Asia/Jakarta. Using UTC." in caplog.text
    # assert "Using UTC." in caplog.text
    assert mock_now.call_count == 2
    mock_now.assert_any_call(tz="UTC")
    mock_timestamp_error.tz_convert.assert_called_once_with("Asia/Jakarta")


def test_filter_out_invalid_entries(caplog):
    df_input = pd.DataFrame({
        "title": ["Product A", "Unknown Product", "Product B", "unknown product C"],
        "price": [10, 50, 20, 5],
    })
    with caplog.at_level(logging.INFO):
        df_filtered = _filter_invalid_entries(df_input.copy())

    expected_filtered_df = pd.DataFrame({
        "title": ["Product A", "Product B"],
        "price": [10, 20],
    }, index=[0, 2])

    assert_frame_equal(df_filtered, expected_filtered_df)
    assert "Filtered out 2 rows with the title 'Unknown Product'." in caplog.text


def test_filter_no_invalid_entries():
    input_df = pd.DataFrame({
        "title": ["Product A", "Product B"],
        "price": [10, 20],
    })
    filtered_df = _filter_invalid_entries(input_df.copy())
    assert_frame_equal(filtered_df, input_df)


def test_filter_invalid_entries_empty_dataframe():
    input_df = pd.DataFrame({"title": []})
    filtered_df = _filter_invalid_entries(input_df.copy())
    assert filtered_df.empty


def test_conversion_application():
    input_df = pd.DataFrame({"price_usd_value": [10.0, 25.5, None, 50.0]})
    converted_df = _apply_conversion(input_df.copy())

    assert "price_idr" in converted_df.columns
    expected_series = pd.Series(
        [
            10.0 * USD_IDR_EXCHANGE_RATE,
            25.5 * USD_IDR_EXCHANGE_RATE,
            None,
            50.0 * USD_IDR_EXCHANGE_RATE,
        ],
        name="price_idr",
    )
    assert_series_equal(converted_df["price_idr"], expected_series, check_dtype=False)


def test_conversion_application_empty_dataframe():
    input_df = pd.DataFrame({"price_usd_value": []})
    converted_df = _apply_conversion(input_df.copy())
    assert "price_idr" in converted_df.columns
    assert converted_df.empty


def test_finalize_columns_structure():
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    intermediate_df = pd.DataFrame({
        "title": ["Product A", "Product B"],
        "price_usd_value": [10.0, 20.0],
        "price_idr": [160000.0, 320000.0],
        "rating_value": [4.5, 4.0],
        "color_count": [3, 1],
        "size": ["M", "S"],
        "gender": ["Men", "Women"],
        "image_url": ["urlA", "urlB"],
        "timestamp": [current_time, current_time],
        "extra_col": ["x", "y"],
    })

    final_df = _finalize_columns(intermediate_df.copy())
    assert sorted(final_df.columns) == sorted(EXPECTED_COLS)
    assert "extra_col" not in final_df.columns
    assert "price_usd_value" not in final_df.columns
    assert final_df["price"].equals(intermediate_df["price_idr"])
    assert final_df["rating"].equals(intermediate_df["rating_value"])
    assert final_df["colors"].equals(intermediate_df["color_count"])

    for col_name, expected_type in SCHEMA_TYPE_MAP.items():
        if col_name in final_df.columns:
            actual_type = final_df[col_name].dtype
            if expected_type == str:
                assert pd.api.types.is_string_dtype(actual_type) or pd.api.types.is_object_dtype(actual_type), \
                    f"Column '{col_name}' expected str, got {actual_type}"
            elif expected_type == float:
                assert pd.api.types.is_float_dtype(actual_type), \
                    f"Column '{col_name}' expected float, got {actual_type}"
            elif isinstance(expected_type, pd.Int64Dtype):
                assert pd.api.types.is_integer_dtype(actual_type), \
                    f"Column '{col_name}' expected Int64Dtype compatible, got {actual_type}"
            elif col_name == "timestamp":
                assert pd.api.types.is_datetime64_any_dtype(actual_type), \
                    f"Column '{col_name}' expected datetime, got {actual_type}"
            else:
                assert pd.api.types.is_dtype_equal(actual_type, expected_type), \
                    f"Column '{col_name}' expected {expected_type}, got {actual_type}"


def test_finalize_columns_with_int64_dtype_conversion():
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    intermediate_df = pd.DataFrame({
        "title": ["Product A"],
        "price_idr": [10000.0],
        "rating_value": [4.5],
        "color_count": [3],
        "size": ["M"],
        "gender": ["Men"],
        "image_url": ["urlA"],
        "timestamp": [current_time],
    })

    custom_schema = {
        "title": str,
        "price": float,
        "rating": float,
        "colors": pd.Int64Dtype(),
        "size": str,
        "gender": str,
        "image_url": str,
        "timestamp": "datetime64[ns, Asia/Jakarta]",
    }

    with patch("utils.transform.SCHEMA_TYPE_MAP", custom_schema):
        final_df = _finalize_columns(intermediate_df.copy())
    assert isinstance(final_df["colors"].dtype, pd.Int64Dtype)


def test_finalize_columns_with_missing_expected_columns(caplog):
    intermediate_df = pd.DataFrame({
        "title": ["A"],
        "price_idr": [1000.0],
    })
    with caplog.at_level(logging.ERROR):
        final_df = _finalize_columns(intermediate_df.copy())

    assert final_df.empty
    assert "The following expected columns are missing before the final selection." in caplog.text
    # assert "'rating_value'" in caplog.text


def test_finalize_columns_handles_type_conversion_error(caplog):
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    intermediate_df = pd.DataFrame({
        "title": ["Product A"],
        "price_idr": ["not_a_number"],
        "rating_value": [4.5],
        "color_count": [3],
        "size": ["M"],
        "gender": ["Men"],
        "image_url": ["urlA"],
        "timestamp": [current_time],
    })
    with caplog.at_level(logging.ERROR):
        final_df = _finalize_columns(intermediate_df.copy())
    assert not final_df.empty
    assert "price" in final_df.columns
    assert "An error occurred during the final data type conversion for column." in caplog.text
    assert pd.api.types.is_object_dtype(final_df["price"].dtype)


def test_finalize_columns_with_incorrect_timestamp_type():
    intermediate_df = pd.DataFrame({
        "title": ["Product A"],
        "price_idr": [160000.0],
        "rating_value": [4.5],
        "color_count": [3],
        "size": ["M"],
        "gender": ["Men"],
        "image_url": ["urlA"],
        "timestamp": ["2023-10-27 10:00:00+07:00"],
    })
    final_df = _finalize_columns(intermediate_df.copy())
    assert pd.api.types.is_datetime64_any_dtype(final_df["timestamp"])
    assert final_df.loc[0, "timestamp"] == pd.Timestamp("2023-10-27 10:00:00+07:00")


def test_drop_nulls_and_duplicates_handling(caplog):
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    df_with_issues = pd.DataFrame({
        "title": ["A", "B", "C", "A", "D", "E"],
        "price": [10.0, 20.0, 30.0, 10.0, 40.0, 50.0],
        "rating": [4, 5, None, 4, 3, 2],
        "colors": [1, 2, 3, 1, 4, 5],
        "size": ["S", "M", None, "S", "L", "M"],
        "gender": ["M", "F", "M", "M", "F", "F"],
        "image_url": ["url1", "url2", "url3", "url1", None, "url5"],
        "timestamp": [current_time] * 6,
    })
    with patch(
        "utils.transform.EXPECTED_COLUMNS", ["title", "size", "gender", "image_url"]
    ):
        with caplog.at_level(logging.INFO):
            cleaned_df = _drop_nulls_and_duplicates(df_with_issues.copy())

    expected_df = pd.DataFrame({
        "title": ["A", "B", "E"],
        "price": [10.0, 20.0, 50.0],
        "rating": [4.0, 5.0, 2.0],
        "colors": [1, 2, 5],
        "size": ["S", "M", "M"],
        "gender": ["M", "F", "F"],
        "image_url": ["url1", "url2", "url5"],
        "timestamp": [current_time] * 3,
    }, index=[0, 1, 5])

    assert_frame_equal(cleaned_df, expected_df)
    assert "Removed 2 rows containing null values in required columns." in caplog.text
    assert "Removed 1 duplicate rows." in caplog.text


def test_drop_nulls_and_duplicates_when_no_issues():
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    clean_df = pd.DataFrame({
        "title": ["A", "B"],
        "price": [10.0, 20.0],
        "rating": [4, 5],
        "colors": [1, 2],
        "size": ["S", "M"],
        "gender": ["M", "F"],
        "image_url": ["url1", "url2"],
        "timestamp": [current_time] * 2,
    })
    with patch(
        "utils.transform.EXPECTED_COLUMNS", ["title", "size", "gender", "image_url"]
    ):
        result_df = _drop_nulls_and_duplicates(clean_df.copy())
    assert_frame_equal(result_df, clean_df)


def test_drop_nulls_and_duplicates_with_empty_dataframe():
    empty_df = pd.DataFrame(columns=EXPECTED_COLS)
    result_df = _drop_nulls_and_duplicates(empty_df.copy())
    assert result_df.empty


def test_drop_nulls_and_duplicates_results_empty_after_na_removal(caplog):
    current_time = pd.Timestamp.now(tz="Asia/Jakarta")
    df_with_all_na = pd.DataFrame({
        "title": ["A", "B"],
        "price": [10.0, 20.0],
        "rating": [4, 5],
        "colors": [1, 2],
        "size": [None, None],
        "gender": ["M", "F"],
        "image_url": ["url1", "url2"],
        "timestamp": [current_time] * 2,
    })
    with patch("utils.transform.EXPECTED_COLUMNS", ["size"]):
        with caplog.at_level(logging.WARNING):
            result_df = _drop_nulls_and_duplicates(df_with_all_na.copy())
    assert result_df.empty
    assert "he DataFrame is empty after removing rows with null values." in caplog.text


@patch(
    "utils.transform._append_timestamp",
    side_effect=lambda df: df.assign(
        timestamp=pd.Timestamp("2023-01-01", tz="Asia/Jakarta")
    ),
)
@patch(
    "utils.transform.EXPECTED_COLUMNS", ["title", "size", "gender", "image_url"]
)
def test_clean_and_transform_success(mock_append_ts, caplog):
    expected_data = {
        "title": [
            "Hoodie Classic",
            "Running Shorts",
            "Wool Scarf",
        ],
        "price": [
            25.0 * USD_IDR_EXCHANGE_RATE,
            18.99 * USD_IDR_EXCHANGE_RATE,
            19.5 * USD_IDR_EXCHANGE_RATE,
        ],
        "rating": [4.7, 4.2, 4.4],
        "colors": [4, 3, 3],
        "size": ["XL", "M", "XL"],
        "gender": [
            "Men",
            "Unisex",
            "Women",
        ],
        "image_url": [
            "url1",
            "url2",
            "url8",
        ],
        "timestamp": pd.Timestamp("2023-01-01", tz="Asia/Jakarta"),
    }

    expected_data["price"] = [
        float(p) if p is not None else None for p in expected_data["price"]
    ]
    expected_data["rating"] = [
        float(r) if r is not None else None for r in expected_data["rating"]
    ]
    expected_data["colors"] = [
        int(c) if c is not None else None for c in expected_data["colors"]
    ]

    expected_df = pd.DataFrame(expected_data)

    for col, dtype in SCHEMA_TYPE_MAP.items():
        if col in expected_df.columns:
            if dtype == pd.Int64Dtype() and expected_df[col].isna().any():
                expected_df[col] = pd.Series(expected_df[col], dtype=pd.Int64Dtype())
            else:
                try:
                    expected_df[col] = expected_df[col].astype(dtype)
                except:
                    print(f"Warning: Failed to cast column {col} to {dtype}")
                    pass

    with caplog.at_level(logging.INFO):
        transformed_df = clean_and_transform(SAMPLE_RAW_DATA)

    transformed_df = transformed_df.sort_values(by="title").reset_index(drop=True)
    expected_df = expected_df.sort_values(by="title").reset_index(drop=True)

    assert not transformed_df.empty
    assert len(transformed_df) == 3
    assert mock_append_ts.called

    for col in expected_df.columns:
        # print("=== EXPECTED DF ===")
        # print(expected_df)
        # print(expected_df.dtypes)
        #
        # print("=== TRANSFORMED DF ===")
        # print(transformed_df)
        # print(transformed_df.dtypes)
        pd.testing.assert_series_equal(
            transformed_df[col],
            expected_df[col],
            check_dtype=False,
            check_names=False,
        )
    assert "Transformation finished. Resulting rows: 3" in caplog.text


def test_clean_and_transform_empty_input(caplog):
    with caplog.at_level(logging.WARNING):
        result_df = clean_and_transform([])
    assert result_df.empty
    assert "Received an empty list for transformation. Returning an empty DataFrame." in caplog.text


@patch("utils.transform._filter_invalid_entries", return_value=pd.DataFrame())
def test_clean_and_transform_empty_after_filter(mock_filter_invalid, caplog):
    with caplog.at_level(logging.WARNING):
        result_df = clean_and_transform(SAMPLE_RAW_DATA[:1])
    assert result_df.empty
    mock_filter_invalid.assert_called_once()
    assert "The DataFrame is empty after filtering out invalid rows." in caplog.text


@patch("utils.transform._finalize_columns", return_value=pd.DataFrame())
def test_clean_and_transform_empty_after_schema_prep(mock_finalize_cols, caplog):
    with patch("utils.transform._filter_invalid_entries", side_effect=lambda df: df):
        with caplog.at_level(logging.ERROR):
            result_df = clean_and_transform(SAMPLE_RAW_DATA[:1])

    assert result_df.empty
    mock_finalize_cols.assert_called_once()
    assert "Final schema preparation failed because the DataFrame became empty." in caplog.text


def test_clean_and_transform_key_error(caplog):
    invalid_data = [{"price": "$10"}]
    with caplog.at_level(logging.ERROR):
        result_df = clean_and_transform(invalid_data)
    assert result_df.empty
    assert "An expected key is missing during transformation. Please verify the extraction keys." in caplog.text


def test_clean_and_transform_unexpected_error(caplog):
    mock_effects = [Exception("Test unexpected error"), pd.DataFrame()]
    with patch("utils.transform.pd.DataFrame", side_effect=mock_effects) as mock_df:
        with caplog.at_level(logging.ERROR):
            result_df = clean_and_transform(SAMPLE_RAW_DATA[:1])

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    assert "An unexpected error occurred during data transformation" in caplog.text
    assert "Test unexpected error" in caplog.text
    assert mock_df.call_count == 2


def test_clean_and_transform_empty_after_nulls_duplicates(caplog):
    sample_data = [
        {
            "title": "Baseball Cap",
            "price": "$14.00",
            "rating": "⭐ 4.1 / 5",
            "colors": "6 color",
            "size": None,
            "gender": "Unisex",
            "image_url": "url7",
        },
        {
            "title": "Baseball Shirt",
            "price": "$29.00",
            "rating": "⭐ 3.8 / 5",
            "colors": "1 colors",
            "size": None,
            "gender": "Men",
            "image_url": "url8",
        },
    ]
    with patch("utils.transform._filter_invalid_entries", side_effect=lambda df: df):
        with caplog.at_level(logging.WARNING):
            result_df = clean_and_transform(sample_data)

    assert result_df.empty
    assert "The DataFrame is empty after removing rows with null values." in caplog.text