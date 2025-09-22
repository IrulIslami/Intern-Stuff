import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import logging
import pandas as pd

# Konfigurasi logging untuk melihat proses
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_pelita_data_pivoted(max_pages, x,y):
    
    target_indicators = [
        "Jumlah Menara Telekomunikasi",
        "Nama Domain Pemerintah Desa",
        "Perangkat Daerah yang memiliki akses internet",
        "Area publik yang memiliki akses internet yang disediakan oleh Dinas Kominfo",
        "Nama Sub Domain Pemerintah Daerah"
    ]

    urls_to_try = [
        "https://pelita.kemendagri.go.id/pemda/dataset/detail?jenis=komponen&kode=1574&urusan=2.16&kwprov={x}&kwkabkot={x}.{y}&kode_jenis=5",
        "https://pelita.kemendagri.go.id/pemda/dataset/detail?jenis=komponen&kode=1541&urusan=2.16&kwprov={x}&kwkabkot={x}.{y}&kode_jenis=5"
    ]

    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    try:
        driver = webdriver.Chrome(options=options)
        logging.info("Chrome WebDriver berhasil diinisialisasi.")
    except Exception as e:
        logging.error(f"Gagal menginisialisasi ChromeDriver: {e}")
        return

    base_url = None
    logging.info("Mencari URL yang valid...")
    for url in urls_to_try:
        logging.info(f"Mencoba URL: ...kode={url.split('kode=')[1].split('&')[0]}...")
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        if soup.find('table', class_='table-digipay'):
            logging.info("Tabel ditemukan! URL ini akan digunakan untuk scraping.")
            base_url = url
            break
        else:
            logging.warning("Tabel tidak ditemukan di URL ini. Mencoba URL berikutnya...")
    
    if not base_url:
        logging.error("Tidak ada URL yang valid yang mengandung tabel 'table-digipay'. Proses berhenti.")
        driver.quit()
        return

    # --- PERUBAHAN 1: Menggunakan dictionary untuk menampung data yang akan dipivot ---
    processed_data = {}
    
    current_page = 1
    logging.info(f"Memulai proses scraping dengan BATAS {max_pages} HALAMAN...")

    while current_page <= max_pages:
        target_url = f"{base_url}&page={current_page}&perpage=25"
        
        logging.info(f"Mengambil data dari halaman: {current_page}")
        driver.get(target_url)
        time.sleep(5) 
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        table = soup.find('table', class_='table-digipay')
        
        if not table:
            logging.warning(f"Tabel tidak ditemukan di halaman {current_page}.")
            break
            
        rows = table.select('tbody tr')
        
        if not rows:
            logging.info(f"Tidak ada baris data pada halaman {current_page}. Proses scraping selesai.")
            break

        for row in rows:
            cells = [cell.text.strip() for cell in row.find_all('td')]
            # Indeks: 2=URAIAN INDIKATOR, 6=KODE PEMDA, 7=TAHUN, 8=DATA
            if len(cells) > 8:
                indicator_text = cells[2]
                
                if indicator_text in target_indicators:
                    kode_pemda = cells[6]
                    tahun = cells[7]
                    data_value = cells[8]
                    
                    # --- PERUBAHAN 2: Mengisi dictionary berdasarkan KODE PEMDA ---
                    # Jika KODE PEMDA ini belum ada, buat entri baru
                    if kode_pemda not in processed_data:
                        processed_data[kode_pemda] = {'TAHUN': tahun}
                    
                    # Tambahkan data indikator ke KODE PEMDA yang sesuai
                    processed_data[kode_pemda][indicator_text] = data_value
        
        logging.info(f"Halaman {current_page} selesai diproses.")
        current_page += 1

    driver.quit()

    if processed_data:
        logging.info(f"Total data untuk {len(processed_data)} KODE PEMDA berhasil dikumpulkan.")
        
        # --- PERUBAHAN 3: Mengubah dictionary menjadi DataFrame yang sudah dipivot ---
        df = pd.DataFrame.from_dict(processed_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'KODE PEMDA'}, inplace=True)

        # Mengatur urutan kolom agar lebih rapi
        kolom_utama = ['KODE PEMDA', 'TAHUN']
        kolom_indikator = [col for col in df.columns if col not in kolom_utama]
        df = df[kolom_utama + kolom_indikator]
        
        nama_file = 'data_kependudukan_pivoted.csv'
        df.to_csv(nama_file, index=False, encoding='utf-8-sig')
        logging.info(f"Scraping selesai. Data berhasil disimpan ke file '{nama_file}'")
    else:
        logging.error("Proses selesai, namun tidak ada data yang cocok yang berhasil di-scrape.")

if __name__ == '__main__':
    # Ganti angka ini untuk scrape lebih dari satu halaman
    data = pd.read_csv('test_scrap.csv')
    for row in data:
        x = row['x']
        y = row['y']
        scrape_pelita_data_pivoted(max_pages=5, x=x,y=y)