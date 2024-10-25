import pandas as pd
import numpy as np
import random
from datetime import timedelta
import os

def generate_water_supply_failure_data(n_ids=200, n_days=365) -> pd.DataFrame:
    """
    Generates a dataset simulating water supply failure data for various pipe sections over time.
    
    Args:
        n_ids (int): Number of unique pipe sections (default: 200).
        n_days (int): Number of days of measurement data for each section (default: 365).
    
    Returns:
        pd.DataFrame: A DataFrame containing the generated water supply failure data.
    """
    total_records = n_ids * n_days
    unique_ids = np.arange(1, n_ids + 1)
    materials = ['Beton', 'PVC', 'Stal', 'Żeliwo']
    soil_types = ['Gleba piaszczysta', 'Gleba ilasta', 'Gleba gliniasta']
    traffic_categories = ['Lekki', 'Średni', 'Ciężki']

    data = {
        'ID_odcinka': np.repeat(unique_ids, n_days),
        'Data_pomiaru': [],
        'Materiał': np.random.choice(materials, size=total_records),
        'Wiek_rury': np.random.randint(5, 100, size=total_records),
        'Średnica': np.random.randint(100, 500, size=total_records),
        'Długość_odcinka': np.random.randint(50, 500, size=total_records),
        'Ciśnienie': np.round(np.random.uniform(2, 10, size=total_records), 2),
        'Przepływ': np.random.randint(50, 200, size=total_records),
        'pH_wody': np.round(np.random.uniform(6, 9, size=total_records), 2),
        'Rodzaj_gruntu': np.random.choice(soil_types, size=total_records),
        'Kategoria_ruchu': np.random.choice(traffic_categories, size=total_records),
        'Data_ostatniej_awarii': [],
        'Data_prawdziwej_awarii': [pd.NaT] * total_records
    }

    for i in range(n_ids):
        year = random.randint(1900, 2000)
        measurement_dates = [pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=x) for x in range(n_days)]
        first_measurement_date = measurement_dates[0]
        last_failure_date = first_measurement_date - pd.DateOffset(years=random.randint(5, 30))
        
        data['Data_pomiaru'].extend(measurement_dates)
        data['Data_ostatniej_awarii'].extend([last_failure_date] * n_days)

    for i in range(0, total_records, n_days):
        last_measurement_date = data['Data_pomiaru'][i + n_days - 1]
        failure_date = last_measurement_date + pd.DateOffset(days=random.randint(30, 3650))
        data['Data_prawdziwej_awarii'][i + n_days - 1] = failure_date

    df = pd.DataFrame(data)
    return df

def save_data_to_csv(df: pd.DataFrame, filename: str = 'data_failure.csv', output_dir: str = 'Predictions/output') -> None:
    """
    Saves the generated data to a CSV file in the specified output directory.
    
    Args:
        df (pd.DataFrame): The DataFrame containing generated data.
        filename (str): The name of the CSV file (default: 'data_failure.csv').
        output_dir (str): The directory where the file will be saved (default: 'output').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data generated and saved to '{filepath}'")


df = generate_water_supply_failure_data()

save_data_to_csv(df)
