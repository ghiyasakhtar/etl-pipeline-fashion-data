import logging
import time
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup, Tag
from requests.exceptions import RequestException, Timeout
from utils.constants import DELAY_DURATION, TIMEOUT_DURATION, USER_AGENT

# Konfigurasi logging agar mencatat waktu, level, nama file, dan baris
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

def get_product_specs(
    details_container: Tag, product_data: Dict[str, Optional[str]]
) -> None:
    # Mendapatkan semua elemen paragraf di dalam container detail
    detail_paragraphs = details_container.find_all("p")
    found_details = set()

    # Iterasi tiap paragraf untuk mencari spesifikasi produk
    for paragraph in detail_paragraphs:
        text = paragraph.get_text(strip=True)

        # Cek dan ambil rating jika belum ditemukan
        if "Rating:" in text and "rating" not in found_details:
            product_data["rating"] = text.split("Rating:", 1)[-1].strip()
            found_details.add("rating")

        # Cek dan ambil warna jika belum ditemukan
        elif "Colors" in text and "colors" not in found_details:
            product_data["colors"] = text
            found_details.add("colors")

        # Cek dan ambil ukuran jika belum ditemukan
        elif "Size:" in text and "size" not in found_details:
            product_data["size"] = text.split("Size:", 1)[-1].strip()
            found_details.add("size")

        # Cek dan ambil gender jika belum ditemukan
        elif "Gender:" in text and "gender" not in found_details:
            product_data["gender"] = text.split("Gender:", 1)[-1].strip()
            found_details.add("gender")

def crawl_all_pages(base_url: str, max_pages: int) -> List[Dict[str, Optional[str]]]:
    all_products: List[Dict[str, Optional[str]]] = []
    normalized_base_url = base_url.rstrip("/")  # Hapus slash di akhir jika ada

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            current_url = normalized_base_url + "/"
        else:
            current_url = f"{normalized_base_url}/page{page_num}"

        logging.info("Scraping Page %d: %s", page_num, current_url)

        page_data = fetch_product_items(current_url)

        if page_data is None:
            logging.warning("Failed to fetch/process page %d", page_num)  # ✅ Tambahan ini
            logging.error(
                "Unable to fetch or process page %d (%s). Skipping.",
                page_num,
                current_url,
            )
        elif not page_data:
            logging.warning(
                "No products were found on page %d (%s). This may indicate the end of the results.",
                page_num,
                current_url,
            )
            logging.info(f"No products found on page {page_num}")
            all_products.extend(page_data)
        else:
            all_products.extend(page_data)
            logging.info(
                "Accumulated %d products after scraping page %d.",
                len(all_products),
                page_num,
            )

        if page_num < max_pages and page_data is not None:
            time.sleep(DELAY_DURATION)

    logging.info(
        "Finished scraping %d pages.", max_pages
        # max_pages,
        # len(all_products),
    )
    return all_products


def fetch_product_items(url: str) -> Optional[List[Dict[str, Optional[str]]]]:
    products: List[Dict[str, Optional[str]]] = []
    try:
        headers = {"User-Agent": USER_AGENT}
        # Kirim permintaan HTTP dengan timeout
        response = requests.get(url, headers=headers, timeout=TIMEOUT_DURATION)
        response.raise_for_status()
        logging.info("HTML content was successfully retrieved from %s.", url)
    except Timeout:
        # Waktu habis
        logging.error("A timeout occurred while trying to fetch the URL %s", url)
        return None
    except RequestException as e:
        # Gagal dalam permintaan HTTP
        logging.error(f"Request failed for URL {url}: {e}")
        return None

    try:
        # Parse HTML menggunakan BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        collection_list = soup.find("div", id="collectionList")

        if not collection_list:
            # Tidak menemukan div utama daftar produk
            logging.warning(
                f"Could not locate the collection list div with id='collectionList' on page {url}."
            )
            return []

        product_cards = collection_list.find_all("div", class_="collection-card")
        if not product_cards:
            # Tidak menemukan kartu produk
            logging.warning(
                f"Collection list found on page {url}, but it contains no product cards."
            )
            return []

        logging.info(f"Detected {len(product_cards)} product cards on page {url}.")

        # Proses setiap kartu produk
        for card in product_cards:
            product_data = process_product_card(card, url)
            if product_data:
                products.append(product_data)

    except Exception as e:
        # Tangani kesalahan parsing HTML
        logging.error(
            f"An error occurred during HTML parsing on page {url}: {e}",
            exc_info=True,
        )
        return products

    logging.info(
        "Successfully extracted data for %d products from page %s.",
        len(products),
        url,
    )
    return products

def process_product_card(card: Tag, url: str) -> Optional[Dict[str, Optional[str]]]:
    # Template awal data produk
    product_data: Dict[str, Optional[str]] = {
        "title": None,
        "price": None,
        "rating": None,
        "colors": None,
        "size": None,
        "gender": None,
        "image_url": None,
    }

    # Ambil judul produk
    title_tag = card.find("h3", class_="product-title")
    if not title_tag or not title_tag.get_text(strip=True):
        # Lewatkan jika tidak ada judul valid
        logging.warning(
            "No valid product title found for a card on %s. The card will be skipped.",
            url,
        )
        return None
    product_data["title"] = title_tag.get_text(strip=True)

    # Ambil harga produk
    price_tag = card.find("span", class_="price")
    if price_tag:
        product_data["price"] = price_tag.get_text(strip=True)
    else:
        price_unavailable_tag = card.find("p", class_="price")
        if (
            price_unavailable_tag
            and "Price N/A" in price_unavailable_tag.get_text(strip=True)
        ):
            product_data["price"] = "Price N/A"
        else:
            logging.warning(
                "Price for product '%s' not found on %s.",
                product_data["title"],
                url,
            )

    # Ambil URL gambar produk
    img_tag = card.find("img", class_="collection-image")
    if img_tag and img_tag.has_attr("src"):
        product_data["image_url"] = img_tag["src"]
    else:
        logging.warning(
            "Image URL for product '%s' could not be found on %s.",
            product_data["title"],
            url,
        )

    # Ambil detail produk tambahan
    details_container = card.find("div", class_="product-details")
    if details_container:
        get_product_specs(details_container, product_data)
    else:
        logging.warning(
            "The 'product-details' div for product '%s' was not found on %s.",
            product_data["title"],
            url,
        )

    return product_data