import pandas as pd
import networkx as nx
import math
import os
import torch  # Dodano import PyTorch

# Stałe
WATER_DENSITY = 1000  # kg/m^3
GRAVITY = 9.81  # m/s^2

def get_pipe_roughness(material):
    """
    Zwraca wartość chropowatości rury na podstawie materiału.
    """
    if material.lower() == 'steel':
        return 0.000045  # w metrach
    elif material.lower() == 'pvc':
        return 0.0000015  # w metrach
    else:
        return 0.0001  # domyślna wartość

def find_loops(G):
    """
    Znajduje wszystkie podstawowe pętle w grafie skierowanym.
    Konwertuje graf na nieskierowany do znalezienia cykli.
    Następnie mapuje cykle na Pipe_ID.
    """
    # Konwersja grafu na nieskierowany
    undirected_G = G.to_undirected()
    # Znajdowanie podstawowych cykli za pomocą cycle_basis
    node_cycles = nx.cycle_basis(undirected_G)
    print(f"Znaleziono {len(node_cycles)} cykli na podstawie węzłów: {node_cycles}")

    pipe_cycles = []
    for cycle in node_cycles:
        pipes = []
        num_nodes = len(cycle)
        for i in range(num_nodes):
            u = cycle[i]
            v = cycle[(i + 1) % num_nodes]
            # Sprawdzenie czy istnieje krawędź (rura) między u i v
            if G.has_edge(u, v):
                pipe_id = G[u][v]['id']
            elif G.has_edge(v, u):
                pipe_id = G[v][u]['id']
            else:
                pipe_id = None
            if pipe_id:
                pipes.append(pipe_id)
        # Dodanie cyklu do listy tylko jeśli zawiera przynajmniej jedną rurę
        if pipes:
            pipe_cycles.append(pipes)

    # Debugowanie: Wyświetl znalezione cykle
    print(f"Znaleziono {len(pipe_cycles)} cykli na podstawie rur: {pipe_cycles}")
    return pipe_cycles


def hardy_cross_iteration(flows, pressure_losses, loops):
    """
    Implementuje jedną iterację metody Hardy'ego-Crossa.
    """
    adjustments = []
    for loop in loops:
        delta_P = 0.0
        for pipe in loop:
            direction = 1.0 if flows[pipe] > 0 else -1.0
            delta_P += pressure_losses[pipe] * direction
        adjustment = -delta_P / len(loop) if len(loop) > 0 else 0.0
        adjustments.append(adjustment)
        for pipe in loop:
            flows[pipe] += adjustment
    return adjustments

def calculate_pressure_loss_pytorch(flows, diameters, lengths, epsilons):
    """
    Oblicza straty ciśnienia przy użyciu PyTorch na GPU.
    
    :param flows: Tensor przepływów [num_hours, num_pipes]
    :param diameters: Tensor średnic rur [num_pipes]
    :param lengths: Tensor długości rur [num_pipes]
    :param epsilons: Tensor chropowatości rur [num_pipes]
    :return: Tensor strat ciśnienia [num_hours, num_pipes]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flows = flows.to(device)
    diameters = diameters.to(device)
    lengths = lengths.to(device)
    epsilons = epsilons.to(device)
    
    # Obliczanie prędkości przepływu
    velocity = flows / (math.pi * (diameters / 2.0) ** 2)
    
    # Obliczanie liczby Reynoldsa
    viscosity = 1e-6  # m^2/s
    Re = (velocity * diameters) / viscosity
    
    # Obliczanie współczynnika tarcia
    f = torch.where((Re < 2000.0) & (Re > 0),
                    64.0 / Re,
                    torch.where(Re >= 2000.0, torch.tensor(0.02, device=device), torch.zeros_like(Re)))
    
    # Obliczanie straty ciśnienia
    pressure_loss = f * (lengths / diameters) * (WATER_DENSITY * velocity ** 2) / 2.0
    
    # Przeniesienie wyników z GPU na CPU
    pressure_loss = pressure_loss.cpu()
    
    return pressure_loss

def calculate_flows_hardy_cross(G, demands, main_flows, hour, max_iterations=100, tolerance=1e-3):
    """
    Oblicza przepływy hydrauliczne dla danej godziny za pomocą metody Hardy'ego-Crossa.
    """
    print(f"Rozpoczynanie obliczeń dla godziny {hour}...")
    # Inicjalizacja przepływów
    flows = {}
    
    # Przypisanie przepływów dla P_main_1
    for pipe_id, flow in main_flows.items():
        flows[pipe_id] = flow.get(hour, 0.0)
    
    # Przypisanie przepływów dla P_building
    for building_num, demand in demands.items():
        # Znajdź rurę prowadzącą do budynku
        for u, v, data in G.edges(data=True):
            pipe_id = data['id']
            if pipe_id.endswith(f'_{building_num}'):
                flows[pipe_id] = demand.get(hour, 0.0)
    
    # Znajdowanie cykli
    print(f"Znajdowanie cykli w sieci dla godziny {hour}...")
    # Debugowanie: Wyświetl wszystkie krawędzie w grafie
    print("Lista krawędzi w grafie:")
    for u, v, data in G.edges(data=True):
        print(f"{u} -> {v}, Pipe_ID: {data['id']}")

    loops = find_loops(G)
    
    if not loops:
        print(f"Brak cykli w sieci dla godziny {hour}. Przerywam iteracje.")
        return flows
    
    # Iteracyjne dostosowywanie przepływów
    for iteration in range(max_iterations):
        print(f"Iteracja {iteration+1} dla godziny {hour}...")
        # Przygotowanie danych do GPU przy użyciu PyTorch
        pipe_ids = list(flows.keys())
        flows_list = [flows[pipe_id] for pipe_id in pipe_ids]
        diameters = [G.edges[u, v]['diameter'] for u, v, d in G.edges(data=True) if d['id'] in pipe_ids]
        lengths = [G.edges[u, v]['length'] for u, v, d in G.edges(data=True) if d['id'] in pipe_ids]
        materials = [G.edges[u, v]['material'] for u, v, d in G.edges(data=True) if d['id'] in pipe_ids]
        
        # Konwersja danych na tensory PyTorch
        flows_tensor = torch.tensor(flows_list, dtype=torch.float64)
        diameters_tensor = torch.tensor(diameters, dtype=torch.float64)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float64)
        epsilons_tensor = torch.tensor([get_pipe_roughness(mat) for mat in materials], dtype=torch.float64)
        
        # Obliczanie strat ciśnienia przy użyciu PyTorch
        print(f"Obliczanie strat ciśnienia na GPU dla godziny {hour}, iteracja {iteration+1}...")
        pressure_losses = calculate_pressure_loss_pytorch(flows_tensor.unsqueeze(0), diameters_tensor, lengths_tensor, epsilons_tensor)[0]
        print(f"Straty ciśnienia obliczone dla godziny {hour}, iteracja {iteration+1}.")
        
        # Aktualizacja strat ciśnienia w słowniku
        pressure_losses_dict = {pipe_id: pressure_losses[i].item() for i, pipe_id in enumerate(pipe_ids)}
        
        # Wykonaj iterację Hardy'ego-Crossa
        adjustments = hardy_cross_iteration(flows, pressure_losses_dict, loops)
        
        # Sprawdzenie zbieżności
        if all(math.fabs(adj) < tolerance for adj in adjustments):
            print(f"Zbieżność osiągnięta po {iteration+1} iteracji dla godziny {hour}.")
            break
        if iteration % 10 == 0:
            print(f"Postęp: {iteration+1} iteracji dla godziny {hour}")
    else:
        print(f"Nie osiągnięto zbieżności po {max_iterations} iteracjach dla godziny {hour}.")
    
    # Aktualizacja przepływów w grafie
    for pipe_id in flows:
        # Znajdź krawędź o danym Pipe_ID
        for u, v, data in G.edges(data=True):
            if data['id'] == pipe_id:
                G.edges[u, v]['flow'] = flows[pipe_id]
                break
    
    print(f"Przepływy dla godziny {hour} zostały obliczone.")
    return flows

def save_flow_results(G, filename, include_p_main=True, main_pipe_id='P_main_1'):
    """
    Zapisuje wyniki przepływów do pliku CSV.
    
    :param G: Graf sieci rur
    :param filename: Ścieżka do pliku wyjściowego
    :param include_p_main: Jeśli True, zapisuje wszystkie rury, ale inne P_main_x ustaw na 0
    :param main_pipe_id: Pipe_ID głównej rury (np. 'P_main_1')
    """
    flow_results = []
    for u, v, data in G.edges(data=True):
        pipe_id = data['id']
        # Warunek zapisywania
        if include_p_main:
            # Zapisz wszystkie rury, ale inne P_main_x ustaw na 0
            if pipe_id.startswith('P_main_') and pipe_id != main_pipe_id:
                flow = 0.0
            else:
                flow = data['flow']
        else:
            # Zapisz tylko P_main_1 i jej rozgałęzienia s1
            if pipe_id == main_pipe_id or 's1' in pipe_id:
                flow = data['flow']
            else:
                continue  # Pomijamy inne rury
        
        for hour in range(24):
            flow_results.append({
                'Pipe_ID': pipe_id,
                'Start_Node': u,
                'End_Node': v,
                'Hour': hour,
                'Flow_m3_h': flow if isinstance(flow, (int, float)) else 0.0
            })
    
    # Konwersja do DataFrame
    flow_df = pd.DataFrame(flow_results)
    # Zapis do pliku CSV
    flow_df.to_csv(filename, index=False)
    print(f"Wyniki zostały zapisane do pliku: {filename}")

def prepare_flows_for_batch(all_flows, unique_pipe_ids, batch_hours, main_flows):
    """
    Przygotowuje przepływy dla danego batch'a godzin.
    
    :param all_flows: Słownik przepływów {pipe_id: {hour: flow}}
    :param unique_pipe_ids: Lista unikalnych Pipe_ID do przetworzenia
    :param batch_hours: Lista godzin w batch'u
    :param main_flows: Słownik przepływów dla głównych rur {pipe_id: {hour: flow}}
    :return: Lista przepływów dla batch'a [godzina][rura]
    """
    flows_batch = []
    for hour in batch_hours:
        print(f"Przygotowywanie przepływów dla godziny {hour}...")
        flows = []
        for pipe_id in unique_pipe_ids:
            # Jeśli pipe_id jest w main_flows, używamy znanej wartości
            if pipe_id in main_flows:
                flows.append(main_flows[pipe_id].get(hour, 0.0))
            else:
                # Rozgałęzienia s1 lub inne rury
                flows.append(all_flows.get(pipe_id, {}).get(hour, 0.0))
        flows_batch.append(flows)
    return flows_batch


def main():
    # Definicja katalogu danych
    data_dir = 'FlowAlgorithm'

    # Lista wymaganych plików
    required_files = [
        os.path.join(data_dir, 'hourly_demand.csv'),
        os.path.join(data_dir, 'node_network.csv'),
        os.path.join(data_dir, 'pipes_network.csv'),
        os.path.join(data_dir, 'main_flows.csv')  # Nowy plik zawierający przepływy dla P_main_1
    ]

    # Sprawdzenie, czy wszystkie pliki istnieją
    for file in required_files:
        if not os.path.exists(file):
            print(f"Brak pliku: {file}")
            return

    # Wczytanie danych
    try:
        print("Wczytywanie plików CSV...")
        demand_df = pd.read_csv(os.path.join(data_dir, 'hourly_demand.csv'))
        nodes_df = pd.read_csv(os.path.join(data_dir, 'node_network.csv'))
        pipes_df = pd.read_csv(os.path.join(data_dir, 'pipes_network.csv'))
        main_flows_df = pd.read_csv(os.path.join(data_dir, 'main_flows.csv'))
        print("Pliki CSV zostały wczytane pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas wczytywania plików CSV: {e}")
        return

    # Filtracja rur: tylko P_main_1 i rury zawierające 's1'
    print("Filtracja rur: uwzględnienie tylko P_main_1 i rur z 's1'...")
    filtered_pipes_df = pipes_df[(pipes_df['Pipe_ID'] == 'P_main_1') | (pipes_df['Pipe_ID'].str.contains('s1'))]
    print(f"Zostało przefiltrowanych {len(filtered_pipes_df)} rur.")

    # Zbieranie unikalnych węzłów z przefiltrowanych rur
    relevant_nodes = set(filtered_pipes_df['Start_Node']).union(set(filtered_pipes_df['End_Node']))
    print(f"Znaleziono {len(relevant_nodes)} unikalnych węzłów do dodania do grafu.")

    # Filtrowanie nodes_df do tylko tych węzłów
    filtered_nodes_df = nodes_df[nodes_df['Node_ID'].isin(relevant_nodes)]
    print(f"Zostało przefiltrowanych {len(filtered_nodes_df)} węzłów do grafu.")

    # Tworzenie grafu skierowanego
    G = nx.DiGraph()

    # Dodawanie węzłów do grafu
    print("Dodawanie węzłów do grafu...")
    for _, row in filtered_nodes_df.iterrows():
        node_id = row['Node_ID']
        node_type = row.get('Node_Type', 'Unknown')  # Przypisz typ węzła lub domyślnie 'Unknown'
        demand = row.get('Demand', 0.0)  # Przypisz zapotrzebowanie lub domyślnie 0.0 dla junctions/sources
        G.add_node(node_id, type=node_type, demand=demand)
    print(f"Zostało dodanych {G.number_of_nodes()} węzłów.")


    # Dodawanie krawędzi (rur) do grafu
    print("Dodawanie rur do grafu...")
    for _, row in filtered_pipes_df.iterrows():
        pipe_id = row['Pipe_ID']
        start_node = row['Start_Node']
        end_node = row['End_Node']
        try:
            diameter = float(row['Diameter'])
            length = float(row['Length'])
            material = row['Material']
            # Filtracja: już dokonana
            G.add_edge(start_node, end_node, id=pipe_id, diameter=diameter, length=length, material=material, flow=0.0)
        except ValueError:
            print(f"Niepoprawne dane dla rury {pipe_id}. Pomijanie tej rury.")
            continue
    print(f"Zostało dodanych {G.number_of_edges()} rur do grafu.")

    # Przypisanie przepływów głównych rur P_main_1 z pliku main_flows.csv
    print("Przypisywanie przepływów głównych rur...")
    main_flows = {}
    main_pipe_id = 'P_main_1'
    main_flows_df_filtered = main_flows_df[main_flows_df['Pipe_ID'] == main_pipe_id]
    for _, row in main_flows_df_filtered.iterrows():
        pipe_id = row['Pipe_ID']
        flows = {hour: float(row[f'Hour_{hour}']) for hour in range(24)}
        main_flows[pipe_id] = flows
    print(f"Przypisano przepływy dla {len(main_flows)} rur głównych.")

    # Ustawienie indeksu dla demand_df
    demand_df.set_index('Building', inplace=True)

    # Przypisywanie zapotrzebowania do budynków
    print("Przypisywanie zapotrzebowania do budynków...")
    demands = {}
    # Lista budynków w grafie - węzły z typem 'Building'
    building_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'Building']
    print(f"Znaleziono {len(building_nodes)} budynków w grafie.")

    # Przypisanie zapotrzebowania do każdego budynku na podstawie wczytanych danych
    for node in building_nodes:
        demand = G.nodes[node].get('demand', 0.0)  # Pobierz wartość zapotrzebowania z atrybutów węzła
        building_num = node.split('_')[-1]  # Zakładamy, że numer budynku jest na końcu nazwy węzła, np. Building_123
        demands[building_num] = {hour: demand for hour in range(24)}  # Załóżmy, że zapotrzebowanie jest stałe w każdej godzinie

    print(f"Przypisano zapotrzebowanie dla {len(demands)} budynków.")


    # Wyświetlenie podsumowania brakujących budynków
    missing_buildings = [b for b, d in demands.items() if all(v == 0.0 for v in d.values())]
    if missing_buildings:
        num_missing = len(missing_buildings)
        print(f"Brak danych zapotrzebowania dla {num_missing} budynków.")
        print("Przykłady brakujących budynków:", missing_buildings[:10])

    # Przygotowanie listy unikalnych Pipe_ID do przetworzenia (P_main_1 i jej s1 rozgałęzienia)
    print("Przygotowanie listy unikalnych rur do przetworzenia...")
    pipe_ids_main = list(main_flows.keys())
    # Znajdź wszystkie rozgałęzienia s1
    s1_pipes = list(filtered_pipes_df[filtered_pipes_df['Pipe_ID'].str.contains('s1')]['Pipe_ID'])
    unique_pipe_ids = list(set(pipe_ids_main + s1_pipes))
    unique_pipe_ids.sort()
    print(f"Przygotowano listę {len(unique_pipe_ids)} unikalnych rur do przetworzenia.")

    # Przygotowanie parametrów rur
    diameters = [G.edges[u, v]['diameter'] for u, v, d in G.edges(data=True) if d['id'] in unique_pipe_ids]
    lengths = [G.edges[u, v]['length'] for u, v, d in G.edges(data=True) if d['id'] in unique_pipe_ids]
    materials = [G.edges[u, v]['material'] for u, v, d in G.edges(data=True) if d['id'] in unique_pipe_ids]

    print("Przygotowano parametry rur (średnica, długość, materiał).")

    # Inicjalizacja struktury do przechowywania przepływów
    all_flows = {pipe_id: {hour: 0.0 for hour in range(24)} for pipe_id in main_flows.keys()}
    for pipe_id in main_flows.keys():
        for hour in range(24):
            all_flows[pipe_id][hour] = main_flows[pipe_id][hour]

    # Zainicjalizuj przepływy dla budynków
    print("Przypisywanie przepływów do budynków...")
    for building_num, demand in demands.items():
        for hour in range(24):
            # Znajdź rurę prowadzącą do budynku
            for u, v, data in G.edges(data=True):
                pipe_id = data['id']
                if pipe_id.endswith(f'_{building_num}'):
                    if pipe_id not in all_flows:
                        all_flows[pipe_id] = {}
                    all_flows[pipe_id][hour] = demand[hour]
    print("Przepływy do budynków zostały przypisane.")

    # Ustawienie rozmiaru batch'a
    batch_size = 4  # Możesz dostosować tę wartość w zależności od zasobów GPU

    # Przygotowanie listy godzin do przetworzenia
    hours = list(range(24))

    # Podział godzin na batch'e
    batches = [hours[i:i + batch_size] for i in range(0, len(hours), batch_size)]
    print(f"Podzielono godziny na {len(batches)} batch'e po {batch_size} godzin każda (ostatni batch może być mniejszy).")

    # Przetwarzanie batch'e sekwencyjnie
    for batch_num, batch_hours in enumerate(batches, start=1):
        print(f"\nPrzetwarzanie batch'a {batch_num}: Godziny {batch_hours}")
        
        # Przygotowanie przepływów dla batch'a
        flows_batch = prepare_flows_for_batch(all_flows, unique_pipe_ids, batch_hours, main_flows)
        print("Przygotowano przepływy dla batch'a.")
        
        # Obliczanie strat ciśnienia przy użyciu PyTorch
        print("Obliczanie strat ciśnienia na GPU przy użyciu PyTorch...")
        pressure_losses_batch = calculate_pressure_loss_pytorch(
            torch.tensor(flows_batch, dtype=torch.float64),
            torch.tensor(diameters, dtype=torch.float64),
            torch.tensor(lengths, dtype=torch.float64),
            torch.tensor([get_pipe_roughness(mat) for mat in materials], dtype=torch.float64)
        )
        print("Straty ciśnienia zostały obliczone.")
        
        # Przetwarzanie przepływów dla każdej godziny w batch'u
        for i, hour in enumerate(batch_hours):
            print(f"Przetwarzanie Hardy'ego-Cross dla godziny {hour} w batch'u {batch_num}")
            calculate_flows_hardy_cross(G, demands, main_flows, hour)
            print(f"Przepływy dla godziny {hour} zostały obliczone.")


    # Aktualizacja przepływów w grafie na podstawie wszystkich godzin
    print("\nAktualizacja przepływów w grafie...")
    for pipe_id in unique_pipe_ids:
        total_flow = 0.0
        for hour in hours:
            total_flow += all_flows.get(pipe_id, {}).get(hour, 0.0)
        # Średni przepływ
        found = False
        for u, v, data in G.edges(data=True):
            if data['id'] == pipe_id:
                G.edges[u, v]['flow'] = total_flow / len(hours) if len(hours) > 0 else 0.0
                found = True
                break
        if not found:
            print(f"Nie znaleziono rury o Pipe_ID: {pipe_id}")
    print("Przepływy zostały zaktualizowane w grafie.")

    output_dir = 'FlowAlgorithm'

    # Zapisywanie wyników tylko dla P_main_1 i jej rozgałęzień s1
    print("\nZapisuję wyniki przepływów do pliku CSV (P_main_1 i s1 rozgałęzienia)...")
    output_file_main = os.path.join(output_dir, 'flow_results_source_1.csv')
    save_flow_results(G, output_file_main, include_p_main=False, main_pipe_id=main_pipe_id)

    print("Program zakończył działanie.")

if __name__ == '__main__':
    main()
