# Dicoding: Membangun ETL Pipeline Sederhana

Proyek ini membangun pipeline ETL (Extract, Transform, Load) untuk mengambil data produk dari situs [Fashion Studio Dicoding](https://fashion-studio.dicoding.dev).

## Fitur Utama
- **Scraping Data:** Mengambil informasi produk dari beberapa halaman web.
- **Transformasi:** Pembersihan teks, konversi harga USD ke IDR, penambahan timestamp, penghapusan duplikat.
- **Pemuatan:** Menyimpan data ke CSV, Google Sheets, atau PostgreSQL.
- **Konfigurasi:** Gunakan file `.env` untuk pengaturan.
- **Logging & Testing:** Logging proses dan pengujian dengan `pytest`.

## Struktur Folder
**my_submission/**  
├── **main.py** — Skrip utama  
├── **requirements.txt** — Daftar dependensi   
├── **output.csv** — Output hasil ETL   
├── **google-sheets-api.json** — Kredensial API Google Sheets (Example/Dummy)
├── **.env** — Environment Variables (Example/Dummy)  
├── **utils/** — Modul ETL  
│   ├── **extract.py** — Modul ekstraksi data  
│   ├── **transform.py** — Modul transformasi data  
│   └── **load.py** — Modul loading data  
├── **tests/** — Unit test  
│   ├── **test_extract.py** — Test untuk ekstraksi  
│   └── **test_transform.py** — Test untuk transformasi   
└── **README** — Dokumentasi proyek
