import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
import time
import logging

# Konfigurasi logging untuk melihat proses
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_pelita_data_final(max_pages): 
   
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

    base_url = "https://pelita.kemendagri.go.id/dataset/257/tabel-data?page="
    current_page = 1
    all_data = []

    logging.info(f"Memulai proses scraping dengan BATAS {max_pages} HALAMAN...")

    # --- PERUBAHAN UTAMA DI SINI ---
    while current_page <= max_pages:
        target_url = f"{base_url}{current_page}"
        
        # Hanya perlu get URL jika navigasi bukan dari klik 'next'
        if driver.current_url != target_url:
             logging.info(f"Mengambil data dari halaman: {current_page}")
             driver.get(target_url)

        time.sleep(5) 
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        table = soup.find('table', class_='table-digipay')
        
        if not table:
            logging.error(f"Tabel 'table-digipay' tidak ditemukan di halaman {current_page}. Proses berhenti.")
            break
            
        rows = table.select('tbody tr')
        
        page_has_data = False
        for row in rows:
            if row.find('td', attrs={'data-th': 'tahun'}):
                cells = row.find_all('td')
                if len(cells) >= 6:
                    data_row = [cell.text.strip() for cell in cells]
                    all_data.append(data_row[:6])
                    page_has_data = True
        
        if page_has_data:
            logging.info(f"Berhasil mengekstrak data dari halaman {current_page}. Total data sekarang: {len(all_data)}")
        else:
            logging.warning(f"Tidak ditemukan baris data pada halaman {current_page}, proses dihentikan.")
            break

        # Cek jika ini adalah halaman terakhir yang diizinkan sebelum klik next
        if current_page == max_pages:
            logging.info(f"Batas maksimum {max_pages} halaman telah tercapai. Menghentikan proses.")
            break
            
        try:
            logging.info("Mencari tombol 'Next' untuk pindah halaman...")
            next_button_li = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next")

            if "ant-pagination-disabled" in next_button_li.get_attribute("class"):
                logging.info("Tombol 'next' tidak aktif. Ini halaman terakhir.")
                break

            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", next_button_li)
            time.sleep(1) 

            driver.execute_script("arguments[0].click();", next_button_li)
            
            logging.info("Berhasil klik 'Next'. Pindah ke halaman berikutnya...")
            current_page += 1

        except NoSuchElementException:
            logging.info("Tombol 'next' tidak ditemukan. Ini adalah halaman terakhir.")
            break
        except Exception as e:
            logging.error(f"Terjadi error saat navigasi halaman: {e}")
            break

    driver.quit()

    if all_data:
        logging.info(f"Total {len(all_data)} baris data berhasil dikumpulkan.")
        df = pd.DataFrame(all_data, columns=[
            'TAHUN', 'SEMESTER', 'PROVINSI', 'KABUPATEN KOTA', 
            'JUMLAH PENDUDUK', 'KEPADATAN PENDUDUK (KM2)'
        ])
        nama_file = 'data_kependudukan_5_halaman.csv' # Ubah nama file agar jelas
        df.to_csv(nama_file, index=False, encoding='utf-8-sig')
        logging.info(f"Scraping selesai. Data berhasil disimpan ke file '{nama_file}'")
    else:
        logging.error("Proses selesai, namun tidak ada data yang berhasil di-scrape.")

if __name__ == '__main__':
    scrape_pelita_data_final(max_pages=1)