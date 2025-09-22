import time
import urllib.parse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re

POI= [
    "Mall",
    "toko emas",
    "toko ritel", 
    "restoran",
    "cafe",
    "supermarket",
    "pasar",
]
KABUPATEN= "Mataram"
PROVINSI= "Nusa Tenggara Barat"


# --- 1. Setup ---
driver = webdriver.Chrome()
search_query = f"{POI} {KABUPATEN}, {PROVINSI}"
url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
driver.get(url)
wait = WebDriverWait(driver, 10) # Define wait object once
results_list = []


def find_and_click_back_button(driver, wait):
    """Tries multiple selectors to find and click the 'back' button."""
    # Possible selectors for the back button
    selectors = [
        'button[jsaction*="back"]'                # Technical, language-agnostic selector
    ]
    
    for selector in selectors:
        try:
            # Wait for the button to be clickable, not just present
            back_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            back_button.click()
            print("Successfully clicked the back button.")
            return # Exit the function if successful
        except TimeoutException:
            # If this selector fails, the loop will just try the next one
            print(f"Selector '{selector}' not found, trying next one...")
            continue
            
    # If all selectors fail, raise an exception
    raise NoSuchElementException("Could not find the back button with any of the attempted selectors.")



try:
    print("Waiting for search results to load...")
    scrollable_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
    
    print("Results container loaded. Starting to scroll to load all POIs...")
    # --- 2. Scroll to Load All POIs in the List ---
    for _ in range(): # Simplified scroll loop
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
        time.sleep(2)
        
    print("Finished scrolling. Found initial POI elements.")
    poi_elements = driver.find_elements(By.CSS_SELECTOR, 'a.hfpxzc')
    num_pois = len(poi_elements)
    print(f"Found {num_pois} POIs to scrape. Starting interactive scraping...")

    # --- 3. Loop, Click, and Scrape Each POI ---
    for i in range(num_pois):
        print(f"\n--- Scraping POI {i + 1} of {num_pois} ---")
        try:
            poi_elements = driver.find_elements(By.CSS_SELECTOR, 'a.hfpxzc')
            if i >= len(poi_elements): continue

            poi_to_click = poi_elements[i]
            poi_name = poi_to_click.get_attribute('aria-label')
            if not poi_name: continue # Skip if element has no name
            
            print(f"Clicking on: {poi_name}")
            poi_to_click.click()

            # Scrape details
            # Address
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.Io6YTe')))
            full_address_element = driver.find_element(By.CSS_SELECTOR, 'div.Io6YTe')
            full_address = full_address_element.text
            print(f"Address found: {full_address}")
            
            # Url
            current_url = driver.current_url
            match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', current_url)
            coordinates = f"{match.group(1)}, {match.group(2)}" if match else "N/A"
            
            print(f"Coordinates: {coordinates}")

            results_list.append({'name': poi_name, 'address': full_address, 'coordinates': coordinates})

            # --- UPDATED: Use the new helper function ---
            find_and_click_back_button(driver, wait)

            # Wait for the main results list to be visible again
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
            time.sleep(1) # Extra pause for stability

        except Exception as e:
            print(f"Error scraping POI {i + 1}: {type(e).__name__}. Skipping.")
            try:
                # Attempt to recover by going back
                find_and_click_back_button(driver, wait)
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div[role="feed"]')))
            except Exception:
                print("Could not recover by going back. Stopping script.")
                break

finally:
    print("\n--- Interactive Scraping Finished ---")
    driver.quit()

# --- 4. Save Results to CSV ---
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv('gmaps_toko_emas_mataram.csv', index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {len(results_list)} POIs to gmaps_{POI}_{KABUPATEN}.csv")
    print(df.head())
else:
    print("No data was scraped successfully.")