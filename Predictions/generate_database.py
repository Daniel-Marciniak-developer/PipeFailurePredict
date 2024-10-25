import pandas as pd
import numpy as np
import random
from datetime import timedelta

def generate_water_supply_failure_data(n_ids=200, n_days=365):
    total_records = n_ids * n_days

    # Unique section IDs
    unique_ids = np.arange(1, 1 + n_ids)

    # Random values for columns
    materials = ['Beton', 'PVC', 'Stal', 'Żeliwo']
    grunty = ['Gleba piaszczysta', 'Gleba ilasta', 'Gleba gliniasta']
    ruch = ['Lekki', 'Średni', 'Ciężki']

    # Data dictionary
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
        'Rodzaj_gruntu': np.random.choice(grunty, size=total_records),
        'Kategoria_ruchu': np.random.choice(ruch, size=total_records),
        'Data_ostatniej_awarii': [],
        'Data_prawdziwej_awarii': [pd.NaT] * total_records  # Initialize with NaT
    }

    # Generate Data_pomiaru and Data_ostatniej_awarii for each ID
    for i in range(n_ids):
        # Fixed year for each ID
        year = random.randint(1900, 2000)
        data_pomiaru = [pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=x) for x in range(n_days)]
        
        # Last failure date (random for each ID, at least 5 years before first measurement)
        first_measurement_date = data_pomiaru[0]
        data_ostatniej_awarii = first_measurement_date - pd.DateOffset(years=random.randint(5, 30))

        # Append Data_pomiaru and Data_ostatniej_awarii
        data['Data_pomiaru'].extend(data_pomiaru)
        data['Data_ostatniej_awarii'].extend([data_ostatniej_awarii] * n_days)

    # Set 'Data_prawdziwej_awarii' to be a random date between 30 days and 10 years after the last day of measurements
    for i in range(0, total_records, n_days):
        last_measurement_date = data['Data_pomiaru'][i + (n_days - 1)]
        # Make the 'Data_prawdziwej_awarii' realistic, between 30 days and 10 years after last measurement date
        data['Data_prawdziwej_awarii'][i + (n_days - 1)] = last_measurement_date + pd.DateOffset(days=random.randint(30, 3650))

    # Create the DataFrame
    df = pd.DataFrame(data)

    return df

# Generate the dataset
df = generate_water_supply_failure_data()

# Optionally, save to CSV
df.to_csv('data_failure.csv', index=False)
print("Data generated and saved to 'data_failure.csv'.")
