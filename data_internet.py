import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import logging

# Konfigurasi logging untuk melihat proses
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DIPERBARUI: Fungsi disederhanakan, tanpa loop paginasi ---
def scrape_single_region(driver, x, y):
    
    target_indicators = [
        "Jumlah Menara Telekomunikasi",
        "Nama Domain Pemerintah Desa",
        "Perangkat Daerah yang memiliki akses internet",
        "Area publik yang memiliki akses internet yang disediakan oleh Dinas Kominfo",
        "Nama Sub Domain Pemerintah Daerah"
    ]

    # --- DIPERBARUI: URL sekarang menyertakan perpage=100 untuk menampilkan semua data ---
    url_templates = [
        f"https://pelita.kemendagri.go.id/pemda/dataset/detail?jenis=komponen&kode=1574&urusan=2.16&kwprov={x}&kwkabkot={x}.{y}&kode_jenis=5&page=1&perpage=100",
        f"https://pelita.kemendagri.go.id/pemda/dataset/detail?jenis=komponen&kode=1541&urusan=2.16&kwprov={x}&kwkabkot={x}.{y}&kode_jenis=5&page=1&perpage=100"
    ]

    logging.info(f"Mencari URL yang valid untuk region x={x}, y={y}...")
    for url in url_templates:
        driver.get(url)
        # Waktu tunggu bisa sedikit lebih lama jika halamannya memuat banyak data
        time.sleep(7) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='table-digipay')

        # Jika tabel ditemukan, langsung proses dan keluar dari loop
        if table:
            logging.info("Tabel ditemukan! Memproses semua data dari halaman...")
            processed_data = {}
            rows = table.select('tbody tr')

            for row in rows:
                cells = [cell.text.strip() for cell in row.find_all('td')]
                if len(cells) > 8:
                    indicator_text = cells[2]
                    if indicator_text in target_indicators:
                        kode_pemda = cells[6]
                        tahun = cells[7]
                        data_value = cells[8]
                        if kode_pemda not in processed_data:
                            processed_data[kode_pemda] = {'TAHUN': tahun}
                        processed_data[kode_pemda][indicator_text] = data_value
            
            if not processed_data:
                logging.warning(f"Tabel ditemukan tetapi tidak ada data indikator yang cocok untuk region x={x}, y={y}.")
                return None

            # Setelah selesai memproses, kembalikan data sebagai DataFrame
            df = pd.DataFrame.from_dict(processed_data, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'KODE PEMDA'}, inplace=True)
            return df
        else:
            logging.warning("Tabel tidak ditemukan di URL ini. Mencoba URL berikutnya...")
    
    # Jika setelah mencoba semua URL tidak ada tabel yang ditemukan
    logging.error(f"Tidak ada URL yang valid yang mengandung data untuk region x={x}, y={y}. Melewatkan.")
    return None


if __name__ == '__main__':
    try:
        regions_to_scrape = pd.read_csv('test_scrap.csv', dtype={'x': str, 'y': str})
        print("Columns found in CSV:", regions_to_scrape.columns)
    except FileNotFoundError:
        logging.error("File 'test_scrap.csv' tidak ditemukan! Harap pastikan file ada di folder yang sama.")
        exit()

    all_regions_data = []

    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    driver = webdriver.Chrome(options=options)

    for index, region_row in regions_to_scrape.iterrows():
        x_val = region_row['x']
        y_val = region_row['y']
        kabupaten_name = region_row['kabupaten']
        
        logging.info(f"--- Memulai Scraping untuk {kabupaten_name} (x={x_val}, y={y_val}) ---")
        
        # --- DIPERBARUI: Panggilan fungsi tidak lagi memerlukan max_pages ---
        region_df = scrape_single_region(driver, x=x_val, y=y_val)
        
        if region_df is not None:
            region_df['NAMA KABUPATEN'] = kabupaten_name
            all_regions_data.append(region_df)

    driver.quit()

    if all_regions_data:
        final_df = pd.concat(all_regions_data, ignore_index=True)
        
        kolom_utama = ['NAMA KABUPATEN', 'KODE PEMDA', 'TAHUN']
        kolom_indikator = [col for col in final_df.columns if col not in kolom_utama]
        final_df = final_df[kolom_utama + kolom_indikator]

        nama_file = 'data_gabungan_semua_region.csv'
        final_df.to_csv(nama_file, index=False, encoding='utf-8-sig')
        logging.info(f"Scraping selesai. Semua data berhasil disimpan ke file '{nama_file}'")
    else:
        logging.error("Proses selesai, namun tidak ada data yang berhasil di-scrape dari semua region.")
        
        
if __name__ == '__main__':
    try:
        # Memaksa kolom 'x' dan 'y' untuk dibaca sebagai string
        regions_to_scrape = pd.read_csv('full_data.csv', dtype={'x': str, 'y': str})
        
        print("Columns found in CSV:", regions_to_scrape.columns)
    except FileNotFoundError:
        logging.error("File 'test_scrap.csv' tidak ditemukan! Harap pastikan file ada di folder yang sama.")
        exit()

    # Siapkan daftar untuk menampung semua hasil DataFrame
    all_regions_data = []

    # Inisialisasi WebDriver HANYA SEKALI
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    driver = webdriver.Chrome(options=options)

    # Menggunakan .iterrows() untuk loop yang benar
    for index, region_row in regions_to_scrape.iterrows():
        x_val = region_row['x']
        y_val = region_row['y']
        # --- BARU: Ambil nama kabupaten dari baris saat ini ---
        kabupaten_name = region_row['kabupaten']
        
        logging.info(f"--- Memulai Scraping untuk {kabupaten_name} (x={x_val}, y={y_val}) ---")
        
        # Panggil fungsi untuk scrape satu region
        region_df = scrape_single_region(driver, x=x_val, y=y_val)
        
        # Jika fungsi mengembalikan data, tambahkan kolom kabupaten dan simpan ke daftar utama
        if region_df is not None:
            # --- BARU: Tambahkan kolom NAMA KABUPATEN ke DataFrame hasil scrape ---
            region_df['NAMA KABUPATEN'] = kabupaten_name
            all_regions_data.append(region_df)

    # Tutup browser setelah semua loop selesai
    driver.quit()

    # Gabungkan semua DataFrame menjadi satu file CSV
    if all_regions_data:
        final_df = pd.concat(all_regions_data, ignore_index=True)
        
        # --- DIPERBARUI: Atur ulang urutan kolom agar NAMA KABUPATEN di depan ---
        kolom_utama = ['NAMA KABUPATEN', 'KODE PEMDA', 'TAHUN']
        kolom_indikator = [col for col in final_df.columns if col not in kolom_utama]
        final_df = final_df[kolom_utama + kolom_indikator]

        nama_file = 'data_gabungan_semua_region.csv'
        final_df.to_csv(nama_file, index=False, encoding='utf-8-sig')
        logging.info(f"Scraping selesai. Semua data berhasil disimpan ke file '{nama_file}'")
    else:
        logging.error("Proses selesai, namun tidak ada data yang berhasil di-scrape dari semua region.")