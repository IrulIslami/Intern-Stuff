import requests
import pandas as pd
import time
import json
import os

# --- Konfigurasi ---
NAMA = "ntb_other_pulau"
INPUT_FILE = f"{NAMA}.csv"
KABUPATEN_COLUMN = "KABUPATEN"
PROVINSI_COLUMN = "PROVINSI"

# Endpoint Overpass API
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# --- Baca DataFrame dari file input ---
try: 
    print(f"Membaca data dari '{INPUT_FILE}'...")
    if INPUT_FILE.endswith('.xlsx'):
        df_input = pd.read_excel(INPUT_FILE)
    else:
        df_input = pd.read_csv(INPUT_FILE)
    
    if KABUPATEN_COLUMN not in df_input.columns or PROVINSI_COLUMN not in df_input.columns:
        print(f"Error: File input harus memiliki kolom '{KABUPATEN_COLUMN}' dan '{PROVINSI_COLUMN}'")
        exit()

    df_input = df_input[[PROVINSI_COLUMN, KABUPATEN_COLUMN]].dropna().drop_duplicates(subset=[KABUPATEN_COLUMN]).copy()
    
except FileNotFoundError:
    print(f"Error: File input '{INPUT_FILE}' tidak ditemukan.")
    exit()
except Exception as e:
    print(f"Terjadi error saat membaca file: {e}")
    exit()    


# --- Fungsi Utama ---
def get_relation_id(area_name: str) -> tuple[int | None, str | None]:
    """
    Mencari ID relasi OpenStreetMap untuk nama wilayah,
    mencoba variasi dengan dan tanpa prefix "Kota" atau "Kabupaten".
    """
    variations = []
    
    # 1. Selalu coba nama asli terlebih dahulu
    variations.append(area_name)
    
    # 2. Jika nama input sudah memiliki prefix, coba juga versi TANPA prefix
    if area_name.startswith("Kota "):
        variations.append(area_name.removeprefix("Kota "))
    elif area_name.startswith("Kabupaten "):
        variations.append(area_name.removeprefix("Kabupaten "))
    # 3. Jika nama input POLOS, coba tambahkan prefix
    else:
        variations.append(f"Kota {area_name}")
        variations.append(f"Kabupaten {area_name}")
        
    name_variations = list(dict.fromkeys(variations))

    for name_variation in name_variations:
        print(f"  -> Mencoba nama: '{name_variation}'...")
        query = f"""
        [out:json][timeout:25];
        relation["name"="{name_variation}"]["admin_level"="5"];
        out ids;
        """
        try:
            response = requests.get(OVERPASS_URL, params={'data': query})
            response.raise_for_status()
            data = response.json()
            if data['elements']:
                return data['elements'][0]['id'], name_variation
        except requests.exceptions.RequestException as e:
            print(f"  -> Error jaringan saat mencari ID untuk '{name_variation}': {e}")
            return None, None
        except json.JSONDecodeError:
            print(f"  -> Gagal decode JSON untuk '{name_variation}'.")
            
    return None, None


def fetch_poi_data(relation_id: int, area_name: str, provinsi_name: str) -> list:
    """Mengambil data POI untuk ID relasi tertentu dan menambahkan nama provinsi."""
    poi_query = f"""
    [out:json][timeout:180];
    relation({relation_id});
    map_to_area;
    (
      nwr[shop](area);
      nwr[amenity](area);
    );
    out geom;
    """
    try:
        response = requests.get(OVERPASS_URL, params={'data': poi_query})
        response.raise_for_status()
        data = response.json()
        
        for feature in data.get('elements', []):
            if 'tags' not in feature:
                feature['tags'] = {}
            feature['tags']['kabupaten_kota'] = area_name
            feature['tags']['provinsi'] = provinsi_name
            
        return data.get('elements', [])
        
    except requests.exceptions.RequestException as e:
        print(f"  -> Error jaringan saat mengambil POI untuk '{area_name}': {e}")
    except json.JSONDecodeError:
        print(f"  -> Gagal decode JSON untuk POI di '{area_name}'.")
    return []


# --- Eksekusi Skrip ---
all_pois = []
status_report = []
failed_to_find = []

print(f"Memulai proses untuk {len(df_input)} wilayah unik...")

for i, row in enumerate(df_input.itertuples(index=False)):
    kabupaten_name = getattr(row, KABUPATEN_COLUMN)
    provinsi_name = getattr(row, PROVINSI_COLUMN)
    
    print(f"\n({i+1}/{len(df_input)}) Memproses: {kabupaten_name}, {provinsi_name}")
    status = {"kabupaten": kabupaten_name, "provinsi": provinsi_name, "relation_id": None, "pois_found": 0, "status": "Failed"}
    
    relation_id, found_name = get_relation_id(kabupaten_name)
    
    if relation_id:
        print(f"  -> Ditemukan Relation ID: {relation_id} (sebagai '{found_name}')")
        status["relation_id"] = relation_id
        
        poi_data = fetch_poi_data(relation_id, kabupaten_name, provinsi_name)
        
        if poi_data:
            print(f"  -> Sukses! Ditemukan {len(poi_data)} POI.")
            all_pois.extend(poi_data)
            status["pois_found"] = len(poi_data)
            status["status"] = "Success"
        else:
            print("  -> Tidak ada POI yang ditemukan untuk tipe yang ditentukan.")
            status["status"] = "Selesai (0 POI)"
    else:
        print(f"  -> TIDAK DAPAT MENEMUKAN RELATION ID untuk '{kabupaten_name}'. Dilewati.")
        status["status"] = "ID Not Found"
        failed_to_find.append(kabupaten_name)
        
    status_report.append(status)
    
    if i < len(df_input) - 1:
        print("  -> Menunggu 7 detik sebelum request berikutnya...")
        time.sleep(7)

print("\n--- Proses Selesai ---")

# --- Pembuatan DataFrame Akhir dan Ekspor ---
if all_pois:
    final_data_list = []
    for item in all_pois:
        row = item.get('tags', {})
        row['id'] = f"{item['type']}/{item['id']}"
        row['type'] = item['type']
        
        if 'lat' in item and 'lon' in item:
            row['latitude'] = item['lat']
            row['longitude'] = item['lon']
        elif 'center' in item:
            row['latitude'] = item['center']['lat']
            row['longitude'] = item['center']['lon']

        final_data_list.append(row)

    df_final = pd.DataFrame(final_data_list)
    
    print("\nMenggabungkan kolom kategori...")
    df_final['category'] = pd.Series(dtype='object')
    
    if 'amenity' in df_final.columns:
        df_final['category'] = df_final['category'].fillna(df_final['amenity'])
    if 'shop' in df_final.columns:
        df_final['category'] = df_final['category'].fillna(df_final['shop'])
    
    # PERBAIKAN: Hapus kolom 'amenity' dan 'shop' yang asli
    cols_to_drop = ['amenity', 'shop']
    df_final = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns])

    preferred_order = ['id', 'provinsi', 'kabupaten_kota', 'name', 'category', 'latitude', 'longitude']
    
    # PERBAIKAN: Pastikan kolom dalam preferred_order ada sebelum mengurutkan
    final_order_cols = [col for col in preferred_order if col in df_final.columns]
    remaining_columns = sorted([col for col in df_final.columns if col not in final_order_cols])
    df_final = df_final[final_order_cols + remaining_columns]

    # PERBAIKAN: Menggunakan f-string dengan benar untuk nama file output
    output_csv = f"{NAMA}_categorized.csv"
    df_final.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Semua POI berhasil disimpan ke '{output_csv}'")
else:
    print("Tidak ada POI yang terkumpul.")

# --- Simpan nama yang gagal ke CSV ---
if failed_to_find:
    print(f"\n{len(failed_to_find)} nama kabupaten/kota tidak dapat ditemukan.")
    failed_df = pd.DataFrame(failed_to_find, columns=['KABUPATEN_NOT_FOUND'])
    # PERBAIKAN: Menggunakan f-string dengan benar untuk nama file output
    failed_output_csv = f'{NAMA}_kabupaten_not_found.csv'
    failed_df.to_csv(failed_output_csv, index=False, encoding='utf-8')
    print(f"❌ Daftar nama yang gagal disimpan ke '{failed_output_csv}'")
else:
    print("\n👍 Semua nama kabupaten/kota berhasil ditemukan!")

# Simpan laporan status
status_df = pd.DataFrame(status_report)
# PERBAIKAN: Menggunakan f-string dengan benar untuk nama file output
status_report_csv = f"{NAMA}_status_report.csv"
status_df.to_csv(status_report_csv, index=False, encoding='utf-8')
print(f"📊 Laporan status lengkap disimpan ke '{status_report_csv}'")