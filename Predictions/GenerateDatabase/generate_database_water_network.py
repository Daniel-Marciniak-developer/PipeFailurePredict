import pandas as pd 
import numpy as np
import random
from datetime import timedelta
import os

def generate_pipe_ids(n_ids=100) -> list:
    """
    Generates a list of unique ID_odcinka using a structured naming convention.
    
    Args:
        n_ids (int): Number of unique ID_odcinka to generate (default: 100).
    
    Returns:
        list: A list containing unique ID_odcinka.
    """
    pipe_ids = []
    sources = ['s1', 's2', 's3']
    pipe_counter = 1

    for source in sources:
        pipe_id_main = f"P_main_{source}"
        pipe_ids.append(pipe_id_main)
        if len(pipe_ids) >= n_ids:
            break

        # Generate secondary branches
        for main_branch in range(1, random.randint(5, 10)):
            pipe_id_secondary = f"P_secondary_{source}_{main_branch}"
            pipe_ids.append(pipe_id_secondary)
            if len(pipe_ids) >= n_ids:
                break

            # Generate tertiary branches
            for second_branch in range(1, random.randint(2, 5)):
                pipe_id_tertiary = f"P_tertiary_{source}_{main_branch}_{second_branch}"
                pipe_ids.append(pipe_id_tertiary)
                if len(pipe_ids) >= n_ids:
                    break

                # Generate building branches
                for building_branch in range(1, random.randint(2, 5)):
                    pipe_id_building = f"P_building_{source}_{main_branch}_{second_branch}_{building_branch}"
                    pipe_ids.append(pipe_id_building)
                    if len(pipe_ids) >= n_ids:
                        break
            if len(pipe_ids) >= n_ids:
                break
        if len(pipe_ids) >= n_ids:
            break

    # If still not enough ID_odcinka, generate additional main loops
    while len(pipe_ids) < n_ids:
        pipe_id_extra = f"P_extra_{pipe_counter}"
        pipe_ids.append(pipe_id_extra)
        pipe_counter += 1

    return pipe_ids[:n_ids]

def generate_water_supply_failure_data(id_odcinka_list: list, n_days=365) -> pd.DataFrame:
    """
    Generates a dataset simulating water supply failure data for various pipe sections over time.
    
    Args:
        id_odcinka_list (list): List of unique ID_odcinka.
        n_days (int): Number of days of measurement data for each section (default: 365).
    
    Returns:
        pd.DataFrame: A DataFrame containing the generated water supply failure data.
    """
    n_ids = len(id_odcinka_list)
    total_records = n_ids * n_days
    
    materials = ['Beton', 'PVC', 'Stal', 'Żeliwo', 'Metal', 'PEHD', 'GRP']
    soil_types = ['Gleba piaszczysta', 'Gleba ilasta', 'Gleba gliniasta']
    traffic_categories = ['Lekki', 'Średni', 'Ciężki']

    # Przygotowanie stałych atrybutów dla każdego ID_odcinka
    pipe_sections = pd.DataFrame({
        'ID_odcinka': id_odcinka_list,
        'Materiał': np.random.choice(materials, size=n_ids),
    })

    # Definiowanie zakresów wieku rury w zależności od materiału
    material_age_ranges = {
        'Stal': (20, 100),
        'PVC': (10, 80),
        'PEHD': (15, 90),
        'GRP': (10, 85),
        'Beton': (30, 120),
        'Żeliwo': (40, 150),
        'Metal': (25, 110)
    }

    # Definiowanie zakresów średnicy w zależności od materiału
    material_diameter_ranges = {
        'Stal': (200, 500),
        'PVC': (100, 400),
        'PEHD': (80, 350),
        'GRP': (90, 300),
        'Beton': (250, 600),
        'Żeliwo': (300, 700),
        'Metal': (150, 450)
    }

    # Definiowanie zakresów długości odcinka w zależności od materiału
    material_length_ranges = {
        'Stal': (1000, 1200),
        'PVC': (100, 500),
        'PEHD': (100, 400),
        'GRP': (100, 450),
        'Beton': (1000, 1500),
        'Żeliwo': (1200, 1600),
        'Metal': (800, 1300)
    }

    # Przypisywanie wieku, średnicy i długości odcinka
    pipe_sections['Wiek_rury'] = pipe_sections['Materiał'].apply(
        lambda mat: random.randint(*material_age_ranges.get(mat, (10, 100)))
    )
    pipe_sections['Średnica'] = pipe_sections['Materiał'].apply(
        lambda mat: random.randint(*material_diameter_ranges.get(mat, (100, 400)))
    )
    pipe_sections['Długość_odcinka'] = pipe_sections['Materiał'].apply(
        lambda mat: random.randint(*material_length_ranges.get(mat, (100, 500)))
    )

    # Definiowanie współczynników ryzyka dla materiałów
    material_risk_factors = {
        'Stal': 1.5,
        'PVC': 1.0,
        'PEHD': 1.2,
        'GRP': 1.1,
        'Beton': 1.3,
        'Żeliwo': 1.4,
        'Metal': 1.3
    }

    # Przypisywanie współczynnika ryzyka dla każdego ID_odcinka
    pipe_sections['Risk_factor'] = pipe_sections['Materiał'].apply(
        lambda mat: material_risk_factors.get(mat, 1.0)
    )

    # Generowanie Data_pomiaru i Data_ostatniej_awarii
    measurement_dates_list = []
    last_failure_dates_list = []
    data_prawdziwej_awarii_list = []
    id_odcinka_repeated = []

    for pipe_id in id_odcinka_list:
        # Losowy rok dla rozpoczęcia pomiarów
        measurement_start_year = random.randint(1970, 2020)
        measurement_dates = [pd.Timestamp(f'{measurement_start_year}-01-01') + pd.Timedelta(days=x) for x in range(n_days)]
        measurement_dates_list.extend(measurement_dates)
        
        # Losowa data ostatniej awarii przed pierwszym pomiarem
        first_measurement_date = measurement_dates[0]
        last_failure_date = first_measurement_date - pd.DateOffset(days=random.randint(365*5, 365*30))
        last_failure_dates_list.extend([last_failure_date] * n_days)
        
        # Powielanie ID_odcinka dla każdego dnia
        id_odcinka_repeated.extend([pipe_id] * n_days)
        
        # Pobieranie atrybutów rury
        pipe_info = pipe_sections.set_index('ID_odcinka').loc[pipe_id]
        age = pipe_info['Wiek_rury']
        diameter = pipe_info['Średnica']
        length = pipe_info['Długość_odcinka']
        risk_factor = pipe_info['Risk_factor']
        last_failure = last_failure_date

        # Obliczanie skali awarii na podstawie wieku, średnicy, długości i współczynnika ryzyka
        # Im większy wiek, średnica, długość oraz ryzyko, tym mniejsza skala (szybsza awaria)
        scale_failure = (age * 0.5) + (diameter * 0.3) + (length * 0.2)
        scale_failure /= risk_factor  # Modyfikacja skali na podstawie ryzyka materiału
        scale_failure *= 5  # Skalowanie, aby umożliwić większą różnorodność czasów do awarii

        # Generowanie liczby dni do awarii z rozkładu Weibulla
        shape_param = 1.2  # Kształt rozkładu Weibulla
        days_to_failure = np.random.weibull(shape_param) * scale_failure
        days_to_failure = max(30, min(7300, int(days_to_failure)))  # Minimalny czas do awarii 30 dni, maksymalny 7300 dni (20 lat)

        # Obliczanie daty prawdziwej awarii (po ostatnim pomiarze)
        failure_date = measurement_dates[-1] + pd.DateOffset(days=days_to_failure)

        # Przypisanie NaT dla wszystkich dni oprócz ostatniego
        data_prawdziwej_awarii_list.extend([pd.NaT] * (n_days - 1))
        # Przypisanie daty awarii dla ostatniego dnia
        data_prawdziwej_awarii_list.append(failure_date)

    # Tworzenie głównego DataFrame z pomiarami
    data = pd.DataFrame({
        'ID_odcinka': id_odcinka_repeated,
        'Data_pomiaru': measurement_dates_list,
        'Materiał': np.repeat(pipe_sections.set_index('ID_odcinka').loc[id_odcinka_list]['Materiał'].values, n_days),
        'Wiek_rury': np.repeat(pipe_sections.set_index('ID_odcinka').loc[id_odcinka_list]['Wiek_rury'].values, n_days),
        'Średnica': np.repeat(pipe_sections.set_index('ID_odcinka').loc[id_odcinka_list]['Średnica'].values, n_days),
        'Długość_odcinka': np.repeat(pipe_sections.set_index('ID_odcinka').loc[id_odcinka_list]['Długość_odcinka'].values, n_days),
        'Ciśnienie': np.round(np.random.normal(loc=5, scale=1.5, size=total_records), 2),  # Realistyczne ciśnienie w barach
        'Przepływ': np.random.randint(50, 200, size=total_records),  # Przepływ w litrach na minutę
        'pH_wody': np.round(np.random.uniform(6, 9, size=total_records), 2),  # pH wody
        'Rodzaj_gruntu': np.random.choice(soil_types, size=total_records),
        'Kategoria_ruchu': np.random.choice(traffic_categories, size=total_records),
        'Data_ostatniej_awarii': last_failure_dates_list,
        'Data_prawdziwej_awarii': data_prawdziwej_awarii_list
    })

    return data

def save_data_to_csv(df: pd.DataFrame, filename: str = 'data_failure.csv', output_dir: str = 'PipeFailurePredict--WaterPrime/Predictions/data') -> None:
    """
    Saves the generated data to a CSV file in the specified output directory.
    
    Args:
        df (pd.DataFrame): The DataFrame containing generated data.
        filename (str): The name of the CSV file (default: 'data_failure.csv').
        output_dir (str): The directory where the file will be saved (default: 'PipeFailurePredict--WaterPrime/Predictions/data').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data generated and saved to '{filepath}'")

def main():
    """
    Main function to generate ID_odcinka, create water supply failure data, and save it to a CSV file.
    """
    n_ids = 100  # Możesz zmienić na 100 lub inną wartość według potrzeb
    n_days = 365
    print(f"Generating {n_ids} ID_odcinka...")
    id_odcinka = generate_pipe_ids(n_ids)
    print(f"Generated {len(id_odcinka)} ID_odcinka.")

    print("Generating water supply failure data...")
    failure_data = generate_water_supply_failure_data(id_odcinka, n_days)
    print(f"Generated failure data with {len(failure_data)} records.")

    print("Saving failure data to CSV...")
    save_data_to_csv(failure_data)
    print("Process completed successfully.")

    # Opcjonalnie, wyświetlenie przykładowych danych
    print("\nPrzykładowe dane awarii:")
    print(failure_data.head(10))  # Wyświetla pierwsze 10 wierszy
    print("\nPrzykładowe dane awarii (ostatni wiersz każdego ID_odcinka):")
    print(failure_data.groupby('ID_odcinka').tail(1).head(10))  # Wyświetla ostatnie wiersze dla pierwszych 10 ID_odcinka

if __name__ == "__main__":
    main()
