# Dicoding: Membangun ETL Pipeline Sederhana

Proyek ini membangun pipeline ETL (Extract, Transform, Load) untuk mengambil data produk dari situs [Fashion Studio Dicoding](https://fashion-studio.dicoding.dev).

## Fitur Utama
- **Scraping Data:** Mengambil informasi produk dari beberapa halaman web.
- **Transformasi:** Pembersihan teks, konversi harga USD ke IDR, penambahan timestamp, penghapusan duplikat.
- **Pemuatan:** Menyimpan data ke CSV, Google Sheets, atau PostgreSQL.
- **Konfigurasi:** Gunakan file `.env` untuk pengaturan.
- **Logging & Testing:** Logging proses dan pengujian dengan `pytest`.

## Struktur Folder
```
**etl-pipeline-fashion-data/**  
├── **main.py**                  # Skrip utama  
├── **requirements.txt**         # Daftar dependensi   
├── **output.csv**               #  Output hasil ETL   
├── **google-sheets-api.json**   # Kredensial API Google Sheets (Example/Dummy)
├── **.env**                     # Environment Variables (Example/Dummy)  
├── **utils/**                   # Modul ETL  
│   ├── **extract.py**           # Modul ekstraksi data  
│   ├── **transform.py**         # Modul transformasi data  
│   └── **load.py**              # Modul loading data  
├── **tests/**                   # Unit test
│   ├── **test_extract.py**      # Test untuk ekstraksi  
│   └── **test_transform.py**    # Test untuk transformasi   
└── **README**                   # Dokumentasi proyek
```

## Cara Setup dan Jalankan Proyek

### 1. Clone repositori
```bash
git clone https://github.com/ghiyasakhtar/etl-pipeline-fashion-data.git
cd etl-pipeline-fashion-data
```

### 2. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Siapkan file konfigurasi
  * Salin file contoh konfigurasi:
  ```bash
  cp .env.example .env
  cp google-sheets-api.example.json google-sheets-api.json
  ```
  * Isi file `.env` dengan konfigurasi sebenarnya (misal database, API keys, dsb).
  * Isi file `google-sheets-api.json` dengan **kredensial Google Sheets** asli.

### 5. Jalankan pipeline ETL
```bash
python main.py
```

## Testing
```bash
python main.py
```
