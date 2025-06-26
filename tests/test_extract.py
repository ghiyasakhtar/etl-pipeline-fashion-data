from utils import extract
from utils.constants import DELAY_DURATION, USER_AGENT
import logging
# import time
from typing import List, Optional
from bs4 import BeautifulSoup, Tag
from requests.exceptions import HTTPError, RequestException, Timeout
from unittest.mock import MagicMock, call, patch
import pytest
import requests

# Contoh HTML tanpa judul produk
SAMPLE_CARD_HTML_NO_TITLE = """
<div class="collection-card">
    <img class="collection-image" src="image5.jpg">
    <!-- There is no <h3> tag with class 'product-title' -->
    <span class="price">$25.99</span>
    <div class="product-details">
        <p>Rating: ⭐ 4.1 / 5</p>
    </div>
</div>
"""

# Contoh HTML tanpa elemen 'product-details'
SAMPLE_CARD_HTML_NO_DETAILS = """
<div class="collection-card">
    <img class="collection-image" src="image4.jpg">
    <h3 class="product-title">Luxury Skirt</h3>
    <span class="price">$65.99</span>
    <!-- There is no div with class 'product-details' -->
</div>
"""

# Contoh HTML dengan informasi lengkap
SAMPLE_CARD_HTML_FULL = """
<div class="collection-card">
    <img class="collection-image" src="image.jpg">
    <h3 class="product-title">Deluxe Short Pants</h3>
    <span class="price"> $27.99 </span>
    <div class="product-details">
        <p>Rating: ⭐ 4.6 / 5</p>
        <p>Available in 5 Colors</p>
        <p>Size: M </p>
        <p>Gender: Unisex</p>
    </div>
</div>
"""

# Contoh HTML tanpa harga produk
SAMPLE_CARD_HTML_NO_PRICE = """
<div class="collection-card">
    <img class="collection-image" src="image2.jpg">
    <h3 class="product-title">Supreme T-Shirt</h3>
    <!-- There is no <span> element with class 'price' -->
    <div class="product-details">
        <p>Rating: ⭐ 3.9 / 5</p>
        <p>Gender: Women</p>
    </div>
</div>
"""

# Contoh HTML dengan harga berupa teks 'Price Unavailable'
SAMPLE_CARD_HTML_PRICE_UNAVAILABLE = """
<div class="collection-card">
    <img class="collection-image" src="image3.jpg">
    <h3 class="product-title">Old Hat</h3>
    <p class="price">Price N/A</p>
    <div class="product-details">
        <p>Rating: ⭐ 4.1 / 5</p>
        <p>3 Colors</p>
        <p>Size: S</p>
    </div>
</div>
"""

# Contoh HTML tanpa gambar produk
SAMPLE_CARD_HTML_NO_IMAGE = """
<div class="collection-card">
    <!-- Tidak ada tag <img> -->
    <h3 class="product-title">Fancy Stripe T-Shirt</h3>
    <span class="price">$67.99</span>
    <div class="product-details">
        <p>Rating: ⭐ 4.7 / 5</p>
    </div>
</div>
"""

SAMPLE_PAGE_HTML_MULTIPLE_CARDS = f"""
<html><body>
<div id="collectionList">
    {SAMPLE_CARD_HTML_FULL}
    {SAMPLE_CARD_HTML_NO_PRICE}
    {SAMPLE_CARD_HTML_PRICE_UNAVAILABLE}
</div>
</body></html>
"""

SAMPLE_PAGE_HTML_EMPTY_LIST = """
<html><body>
<div id="collectionList">
    <!-- There is no collection-card divs -->
</div>
</body></html>
"""

SAMPLE_PAGE_HTML_NO_LIST = """
<html><body>
<div>Some other content</div>
</body></html>
"""

@pytest.mark.parametrize(
    "raw_html, result_dict, predicted_warnings",
    [
        # Test ketika judul produk tidak ditemukan
        (
            SAMPLE_CARD_HTML_NO_TITLE,
            None,
            ["No valid product title found for a card on http://test.com/page1. The card will be skipped."],
        ),

        # Test ketika harga produk tidak tersedia
        (
            SAMPLE_CARD_HTML_NO_PRICE,
            {
                "title": "Supreme T-Shirt",
                "price": None,
                "rating": "⭐ 3.9 / 5",
                "colors": None,
                "size": None,
                "gender": "Women",
                "image_url": "image2.jpg",
            },
            ["Price for product 'Supreme T-Shirt' not found on http://test.com/page1."],
        ),

        # Test ketika gambar produk tidak ditemukan
        (
            SAMPLE_CARD_HTML_NO_IMAGE,
            {
                "title": "Fancy Stripe T-Shirt",
                "price": "$67.99",
                "rating": "⭐ 4.7 / 5",
                "colors": None,
                "size": None,
                "gender": None,
                "image_url": None,
            },
            ["Image URL for product 'Fancy Stripe T-Shirt' could not be found"],
        ),

        # Test lengkap dengan semua informasi produk tersedia
        (
            SAMPLE_CARD_HTML_FULL,
            {
                "title": "Deluxe Short Pants",
                "price": "$27.99",
                "rating": "⭐ 4.6 / 5",
                "colors": "Available in 5 Colors",
                "size": "M",
                "gender": "Unisex",
                "image_url": "image.jpg",
            },
            [],
        ),

        # Test ketika elemen 'product-details' tidak ditemukan
        (
            SAMPLE_CARD_HTML_NO_DETAILS,
            {
                "title": "Luxury Skirt",
                "price": "$65.99",
                "rating": None,
                "colors": None,
                "size": None,
                "gender": None,
                "image_url": "image4.jpg",
            },
            ["The 'product-details' div for product 'Luxury Skirt' was not found on http://test.com/page1."],
        ),

        # Test ketika harga memiliki label "Price N/A"
        (
            SAMPLE_CARD_HTML_PRICE_UNAVAILABLE,
            {
                "title": "Old Hat",
                "price": "Price N/A",
                "rating": "⭐ 4.1 / 5",
                "colors": "3 Colors",
                "size": "S",
                "gender": None,
                "image_url": "image3.jpg",
            },
            [],
        ),
    ],
)

def test_parse_product_card(
    raw_html: str,
    result_dict: Optional[dict],
    predicted_warnings: List[str],
    caplog,
):
    # Parsing HTML input
    parsed_html = BeautifulSoup(raw_html, "html.parser")
    product_card = parsed_html.find("div", class_="collection-card")
    assert isinstance(product_card, Tag)

    test_page_url = "http://test.com/page1"
    caplog.set_level(logging.WARNING)

    # Proses kartu produk
    actual_result = extract.process_product_card(product_card, test_page_url)

    assert actual_result == result_dict

    if predicted_warnings:
        for msg in predicted_warnings:
            assert msg in caplog.text
    else:
        # Pastikan tidak ada peringatan
        for log_record in caplog.records:
            assert log_record.levelno < logging.WARNING


@patch("utils.extract.requests.get")
def test_fetch_product_items_success(mock_get, caplog):
    fake_response = MagicMock(spec=requests.Response)
    fake_response.status_code = 200
    fake_response.text = SAMPLE_PAGE_HTML_MULTIPLE_CARDS
    fake_response.raise_for_status.return_value = None
    mock_get.return_value = fake_response
    url = "http://test.com/page1"
    caplog.set_level(logging.INFO)

    products = extract.fetch_product_items(url)

    mock_get.assert_called_once_with(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=extract.TIMEOUT_DURATION,
    )
    fake_response.raise_for_status.assert_called_once()

    # Pastikan hasil sesuai dan log muncul
    assert products is not None
    assert len(products) == 3
    assert products[0]["title"] == "Deluxe Short Pants"
    assert products[1]["title"] == "Supreme T-Shirt"
    assert products[2]["title"] == "Old Hat"

    assert f"HTML content was successfully retrieved from {url}." in caplog.text
    assert f"Detected 3 product cards on page {url}." in caplog.text
    assert f"Successfully extracted data for 3 products from page {url}" in caplog.text


@patch("utils.extract.requests.get")
def test_fetch_product_items_request_timeout(mock_get, caplog):
    url = "http://timeout.com"
    mock_get.side_effect = Timeout("Request timed out")
    caplog.set_level(logging.ERROR)

    output = extract.fetch_product_items(url)

    # Pastikan None jika timeout terjadi
    assert output is None
    assert f"A timeout occurred while trying to fetch the URL {url}" in caplog.text
    mock_get.assert_called_once_with(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=extract.TIMEOUT_DURATION,
    )


@patch("utils.extract.requests.get")
def test_fetch_product_items_http_error(mock_get, caplog):
    response_mock = MagicMock(spec=requests.Response)
    response_mock.status_code = 404
    response_mock.raise_for_status.side_effect = HTTPError("404 Client Error")
    mock_get.return_value = response_mock
    url = "http://notfound.com"
    caplog.set_level(logging.ERROR)

    result = extract.fetch_product_items(url)

    # Pastikan None dikembalikan saat HTTP error
    assert result is None
    assert f"Request failed for URL {url}: 404 Client Error" in caplog.text
    response_mock.raise_for_status.assert_called_once()


@patch("utils.extract.requests.get")
def test_fetch_product_items_request_fail(mock_get, caplog):
    url = "http://connectionerror.com"
    mock_get.side_effect = RequestException("Connection error")
    caplog.set_level(logging.ERROR)

    output = extract.fetch_product_items(url)

    # Mengembalikan None jika terjadi kesalahan request
    assert output is None
    assert f"Request failed for URL {url}: Connection error" in caplog.text



@patch("utils.extract.requests.get")
def test_fetch_product_items_no_products(mock_get, caplog):
    fake_resp = MagicMock(spec=requests.Response)
    fake_resp.status_code = 200
    fake_resp.text = SAMPLE_PAGE_HTML_EMPTY_LIST  # kosong tanpa produk
    fake_resp.raise_for_status.return_value = None
    mock_get.return_value = fake_resp

    url = "http://test.com/empty-list"
    caplog.set_level(logging.WARNING)

    items = extract.fetch_product_items(url)

    # Pastikan list kosong jika tidak ada produk
    assert items == []
    assert (
            f"Collection list found on page http://test.com/empty-list, but it contains no product cards."
            in caplog.text
    )


@patch("utils.extract.requests.get")
def test_fetch_product_items_empty_collection_list(mock_get, caplog):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.text = SAMPLE_PAGE_HTML_EMPTY_LIST
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    test_url = "http://test.com/empty-list"
    caplog.set_level(logging.WARNING)

    result = extract.fetch_product_items(test_url)

    assert result == []
    assert (
        f"Collection list found on page http://test.com/empty-list, but it contains no product cards."
    ) in caplog.text

@patch("utils.extract.requests.get")
@patch("utils.extract.BeautifulSoup")
def test_product_fetch_parsing_failure(mock_bs, mock_get, caplog):
    fake_resp = MagicMock(spec=requests.Response)
    fake_resp.status_code = 200
    fake_resp.text = "<htm" # Tidak lengkap
    fake_resp.raise_for_status.return_value = None
    mock_get.return_value = fake_resp

    mock_bs.side_effect = Exception("Parsing failed badly")
    url = "http://test.com/parse-error"

    caplog.set_level(logging.ERROR)
    products = extract.fetch_product_items(url)

    # Pastikan fungsi mengembalikan list kosong saat error parsing
    assert products == []
    assert f"An error occurred during HTML parsing on page {url}" in caplog.text
    assert "Parsing failed badly" in caplog.text


@patch("utils.extract.fetch_product_items")
@patch("utils.extract.time.sleep")
def test_all_pages_scrape_complete_flow(mock_sleep, mock_extract, caplog):
    root_url = "http://example.com"
    page_limit = 3

    mock_extract.side_effect = [
        [{"title": "Page1 Prod1"}],
        [{"title": "Page2 Prod1"}, {"title": "Page2 Prod2"}],
        [{"title": "Page3 Prod1"}],
    ]

    caplog.set_level(logging.INFO)
    all_products = extract.crawl_all_pages(root_url, page_limit)

    # Validasi hasil scraping
    assert len(all_products) == 4
    assert all_products[0]["title"] == "Page1 Prod1"
    assert all_products[1]["title"] == "Page2 Prod1"
    assert all_products[3]["title"] == "Page3 Prod1"

    expected_calls = [
        call("http://example.com/"),
        call("http://example.com/page2"),
        call("http://example.com/page3"),
    ]
    mock_extract.assert_has_calls(expected_calls)
    assert mock_extract.call_count == page_limit

    # Cek jumlah jeda yang dilakukan
    assert mock_sleep.call_count == page_limit - 1
    mock_sleep.assert_called_with(DELAY_DURATION)

    # Cek log scraping
    assert "Scraping Page 1: http://example.com/" in caplog.text
    assert "Scraping Page 2: http://example.com/page2" in caplog.text
    assert "Scraping Page 3: http://example.com/page3" in caplog.text
    assert "Accumulated 1 products after scraping page 1." in caplog.text
    assert "Accumulated 3 products after scraping page 2." in caplog.text
    assert "Accumulated 4 products after scraping page 3." in caplog.text
    assert f"Finished scraping {page_limit} pages." in caplog.text


@patch("utils.extract.fetch_product_items")
@patch("utils.extract.time.sleep")
def test_mix_page_conditions(mock_delay, mock_parser, caplog):
    # Inisialisasi URL awal dan jumlah halaman yang ingin ditelusuri
    origin = "http://complex.com/"
    limit = 4

    # Simulasikan hasil parsing dari tiap halaman (berisi, gagal, kosong, berisi)
    mock_parser.side_effect = [
        [{"title": "Page1 Prod1"}],
        None,
        [],
        [{"title": "Page4 Prod1"}],
    ]

    caplog.set_level(logging.INFO)

    # Jalankan proses pengumpulan data
    response = extract.crawl_all_pages(origin, limit)

    # Validasi jumlah item yang berhasil dikumpulkan
    expected_titles = ["Page1 Prod1", "Page4 Prod1"]
    obtained_titles = [item["title"] for item in response]
    assert obtained_titles == expected_titles

    # Periksa pemanggilan terhadap mock_parser
    tracked_urls = [
        call(f"{origin}"),
        *[call(f"{origin}page{i}") for i in range(2, limit + 1)]
    ]
    mock_parser.assert_has_calls(tracked_urls)
    assert mock_parser.call_count == limit

    # Pastikan fungsi delay dipanggil pada kasus gagal/kosong
    assert mock_delay.call_count == 2
    assert mock_delay.call_args_list == [call(DELAY_DURATION)] * 2

    # Cek bahwa log mengandung informasi yang relevan
    assert "Failed to fetch/process page 2" in caplog.text
    assert "No products found on page 3" in caplog.text
    assert "Accumulated 1 products after scraping page 1." in caplog.text
    assert "Accumulated 2 products after scraping page 4." in caplog.text
    assert f"Finished scraping {limit} pages." in caplog.text


@patch("utils.extract.fetch_product_items")
@patch("utils.extract.time.sleep")
def test_single_page_scraping_no_wait(mock_sleep, mock_extract):
    # URL awal untuk scraping
    start_url = "http://single.com"
    # Batasi hanya 1 halaman yang di-scrape
    limit = 1
    # Mock hasil ekstraksi halaman tersebut
    mock_extract.return_value = [{"title": "SinglePageProd"}]

    # Jalankan fungsi crawl_all_pages dengan batas halaman
    scraped_items = extract.crawl_all_pages(start_url, limit)

    # Pastikan hanya 1 item hasil scraping
    assert len(scraped_items) == 1
    # Pastikan fungsi ekstraksi hanya dipanggil sekali dengan URL yang benar
    mock_extract.assert_called_once_with("http://single.com/")
    # Pastikan tidak ada jeda waktu yang dipanggil karena hanya 1 halaman
    mock_sleep.assert_not_called()