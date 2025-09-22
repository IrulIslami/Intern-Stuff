import time
import urllib.parse
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- 1. CONFIGURATION ---
POI_KEYWORDS = [
    "Mall",
    "toko emas",
    "toko ritel", 
    "restoran",
    "cafe",
    "supermarket",
    "pasar",
]


KABUPATEN = input("Masukkan Nama Kabupaten: ")
PROVINSI = input("Masukkan Nama Provinsi: ")

# --- 2. MAIN SCRIPT ---
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()
wait = WebDriverWait(driver, 20)
all_results = []

try:
    for keyword in POI_KEYWORDS:
        print(f"\n========================================================")
        print(f"--- Starting scrape for '{keyword}' in '{KABUPATEN}' ---")
        print(f"========================================================")

        search_query = f"{keyword} di {KABUPATEN}, {PROVINSI}"
        url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
        driver.get(url)

        try:
            scrollable_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
            
            print("Scrolling to load results...")
            for _ in range(25):
                driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
                time.sleep(2)
            
            poi_elements = driver.find_elements(By.CSS_SELECTOR, 'a.hfpxzc[aria-label]')
            poi_links = [elem.get_attribute('href') for elem in poi_elements]
            num_pois = len(poi_links)
            print(f"Found {num_pois} valid POIs to scrape.")
            
            main_tab = driver.current_window_handle

            for i, link in enumerate(poi_links):
                if not link:
                    continue

                print(f"\n--- Scraping POI {i + 1} of {num_pois} ---")
                try:
                    driver.switch_to.new_window('tab')
                    driver.get(link)
                    
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h1')))
                    poi_name = driver.find_element(By.CSS_SELECTOR, 'h1').text
                    print(f"Name: {poi_name}")

                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.Io6YTe')))
                    address = driver.find_element(By.CSS_SELECTOR, 'div.Io6YTe').text
                    print(f"Address: {address}")
                    
                    # --- FIX: Wait for the URL to contain the '@' symbol ---
                    wait.until(EC.url_contains('@'))
                    
                    current_url = driver.current_url
                    match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', current_url)
                    coordinates = f"{match.group(1)}, {match.group(2)}" if match else "N/A"
                    print(f"Coordinates: {coordinates}")

                    all_results.append({
                        'kabupaten': KABUPATEN,
                        'provinsi': PROVINSI,
                        'category': keyword,
                        'name': poi_name, 
                        'address': address, 
                        'coordinates': coordinates
                    })
                    
                    driver.close()
                    driver.switch_to.window(main_tab)
                    time.sleep(1)

                except Exception as e:
                    print(f"Error scraping link {i + 1}: {type(e).__name__}. Skipping.")
                    if len(driver.window_handles) > 1:
                        driver.close()
                    driver.switch_to.window(main_tab)
                    continue
        
        except TimeoutException:
            print(f"No results found for '{keyword}'. Moving to next keyword.")
            continue

finally:
    print("\n--- All Scraping Jobs Finished ---")
    driver.quit()

# --- 3. Save All Results to a Single CSV ---
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(f'gmaps_poi_results_{KABUPATEN}.csv', index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {len(all_results)} total POIs to gmaps_poi_results_{KABUPATEN}.csv")
    print(df.head())
else:
    print("No data was scraped successfully.")