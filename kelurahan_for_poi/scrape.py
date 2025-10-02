import time
import urllib.parse
import pandas as pd
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# df = pd.read_csv('bandung.csv')
# data = pd.read_csv('t.csv')

data = pd.read_csv('gis.dukcapil.kemendagri.xlsx - Batas Desa.csv')

# PROVINSI = input("Masukkan Provinsi: ")
KABUPATEN = input("Masukkan Kabupaten: ")

df = data[data['nama_kab']==KABUPATEN]

num = np.random.randint(1, 3)


# --- 2. MAIN SCRIPT ---
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()
wait = WebDriverWait(driver, 20)
all_results = []

try:
    
    for index, row in df.iterrows():
        
        
        
        search_query = f"{row.nama_kel}, {row.nama_kec}, {row.nama_prop}"
        url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
        driver.get(url)
        
        try:
            
            main_tab = driver.current_window_handle
                        
            wait.until(EC.url_contains('@'))
            current_url = driver.current_url
            match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', current_url)
            coordinates = f"{match.group(1)}, {match.group(2)}" if match else "N/A"
            print(f"Coordinates: {coordinates}")

            all_results.append({
            'objectid':row.objectid,
            'no_prop': row.no_prop,
            'no_kab': row.no_kab,
            'no_kec':	row.no_kec,
            'no_kel':row.no_kel,
            'kode_desa_spatial':row.kode_desa_spatial,
            'provinsi': row.nama_prop,
            'kecamatan': row.nama_kec,
            'kabupaten': row.nama_kab,
            'kelurahan': row.nama_kel,
            'coordinates': coordinates
            })
                    
            time.sleep(num)

        except Exception as e:
            print(f"Error scraping link : {type(e).__name__}. Skipping.")
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_tab)
                continue
        
        except TimeoutException:
            print(f"No results found for '{row.nama_kel}'. Moving to next keyword.")
            continue
        
        print(f"Succes scrap kelurahan {row.nama_kel} with coordinates: {coordinates}")
        
finally:
    print("\n--- All Scraping Jobs Finished ---")
    driver.quit()

# --- 3. Save All Results to a Single CSV ---
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(f'kelurahan_results_{KABUPATEN}.csv', index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {len(all_results)} total POIs to gmaps_poi_results_{KABUPATEN}.csv")
    print(df.head())
else:
    print("No df was scraped successfully.")