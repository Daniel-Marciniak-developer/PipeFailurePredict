import pandas as pd
import numpy as np

# Parametry generowania zapotrzebowania
num_buildings = 9171  # Liczba budynków (B1 do B9995)
hours = 24  # Liczba godzin w ciągu doby

# Ustawienie ziarna losowości dla powtarzalności wyników
np.random.seed(42)

# Generowanie podstawowego zapotrzebowania dla każdego budynku, aby zapewnić różnorodność wzorców
base_demands = np.random.uniform(0.3, 0.5, (num_buildings, hours))

# Generowanie godzinnych wariacji przy użyciu wzorca sinusoidalnego dla cyklu dziennego (poranne/wieczorne szczyty)
hourly_variation = np.sin(np.linspace(0, 2 * np.pi, hours))

# Tworzenie tablicy do przechowywania danych zapotrzebowania
demand_data = []

# Generowanie danych zapotrzebowania dla każdego budynku, zapewniając różnorodność
for building_base in base_demands:
    # Zastosowanie godzinnych wariacji do podstawowego zapotrzebowania budynku
    building_demand = building_base * (1 + hourly_variation)
    # Dodanie unikalnych losowych zakłóceń dla każdego budynku, aby zapewnić różnorodność
    building_demand += np.random.uniform(-0.05, 0.05, hours)
    # Ograniczenie wartości do realistycznego zakresu i zaokrąglenie do 3 miejsc po przecinku
    building_demand = np.clip(building_demand, 0.2, 0.7).round(3)
    demand_data.append(building_demand)

# Konwersja danych zapotrzebowania do DataFrame
columns = ['Building'] + [f'Hour_{i}' for i in range(hours)]
demand_df = pd.DataFrame(demand_data, columns=columns[1:])
demand_df.insert(0, 'Building', [f'B{i}' for i in range(1, num_buildings + 1)])

# Zapisanie danych do pliku CSV
demand_df.to_csv('FlowAlgorithm/hourly_demand.csv', index=False)

# Wyświetlenie pierwszych kilku wierszy wygenerowanej tabeli zapotrzebowania do weryfikacji
print(demand_df.head())
