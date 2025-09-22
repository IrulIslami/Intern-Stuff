import time
import urllib.parse
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

KEY_WORDS= "Toko Emas"
KABUPATEN= "Mataram"
PROVINSI= "Nusa Tenggara Barat"

def find_and_click_back_button(driver, wait):
    """Tries multiple selectors to find and click the 'back' button."""
    selectors = [
        'button[jsaction*="back"]'
    ]
    for selector in selectors:
        try:
            back_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].click();", back_button)
            return
        except TimeoutException:
            continue
    raise NoSuchElementException("Could not find the back button with any of the attempted selectors.")

# --- 1. Setup ---
driver = webdriver.Chrome()
driver.maximize_window()
search_query = f"{KEY_WORDS} {KABUPATEN}, {PROVINSI}"
url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
driver.get(url)
wait = WebDriverWait(driver, 10)
results_list = []

try:
    print("Waiting for search results to load...")
    scrollable_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
    
    # --- 2. Dynamic "Scroll-and-Check" Loop (from the new code) ---
    print("Starting dynamic scroll to load all POIs...")
    last_count = 0
    while True:
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
        time.sleep(2.5)
        current_count = len(driver.find_elements(By.CSS_SELECTOR, 'a.hfpxzc[aria-label]'))
        print(f"Found {current_count} items so far...")
        if current_count == last_count:
            print("Reached the end of the results.")
            break
        last_count = current_count

    driver.execute_script('arguments[0].scrollTop = 0', scrollable_div)
    time.sleep(1)

    # --- 3. Interactive Scraping Loop (structure from your OLD, working code) ---
    num_pois = last_count
    print(f"\nFound a total of {num_pois} POIs. Starting interactive scraping...")

    for i in range(num_pois):
        print(f"\n--- Scraping POI {len(results_list) + 1}/{num_pois} ---")
        try:                
            # Re-find the elements list INSIDE the loop iteration, just like your old code
            poi_elements = driver.find_elements(By.CSS_SELECTOR, 'a.hfpxzc[aria-label]')
            if i >= len(poi_elements):
                print("Index out of bounds, stopping.")
                break

            poi_to_click = poi_elements[i]
            poi_name = poi_to_click.get_attribute('aria-label')
            
            if not poi_name or any(d['name'] == poi_name for d in results_list):
                print(f"Skipping already scraped or invalid item: '{poi_name}'")
                continue
            
            print(f"Clicking on: {poi_name}")
            poi_to_click.click()

            # Scrape details
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.Io6YTe')))
            address = driver.find_element(By.CSS_SELECTOR, 'div.Io6YTe').text
            
            current_url = driver.current_url
            match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', current_url)
            coordinates = f"{match.group(1)}, {match.group(2)}" if match else "N/A"
            
            print(f"Address: {address}")
            print(f"Coordinates: {coordinates}")

            results_list.append({'name': poi_name, 'address': address, 'coordinates': coordinates})
            
            find_and_click_back_button(driver, wait)
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
            time.sleep(1)

        except Exception as e:
            print(f"Error scraping POI at index {i}: {type(e).__name__}. Skipping.")
            try:
                # Attempt to recover by going back
                find_and_click_back_button(driver, wait)
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
            except Exception:
                print("Could not recover. Stopping script.")
                break

finally:
    print("\n--- Scraping Finished ---")
    driver.quit()

# --- 4. Save Results to CSV ---
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv('gmaps_toko_emas_mataram_full.csv', index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {len(results_list)} POIs to gmaps_toko_emas_mataram_full.csv")
    print(df.head())
else:
    print("No data was scraped successfully.")