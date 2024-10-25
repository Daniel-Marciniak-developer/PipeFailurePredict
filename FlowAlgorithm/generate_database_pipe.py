import pandas as pd
import random
import networkx as nx

def generate_pipe_network(building_target):
    pipes_data = []
    pipe_counter = 1

    # Definiujemy źródła
    sources = ['s1', 's2', 's3']
    
    # Tworzymy graf do śledzenia połączeń
    G = nx.Graph()

    # Generowanie trzech głównych rur (main)
    for source_index, source in enumerate(sources, start=1):
        pipe_id = f"P_main_{source_index}"
        start_node = f"Source_{source_index}"
        end_node = f"junction_main_{source}"
        diameter = round(random.uniform(0.25, 0.3), 2)
        length = random.randint(1000, 1200)
        material = 'Steel'
        pipes_data.append([pipe_id, start_node, end_node, diameter, length, material])
        G.add_edge(start_node, end_node, id=pipe_id)

        # Tworzenie drugorzędnych rur (secondary)
        for main_branch in range(1, random.randint(20, 35)):
            pipe_id = f"P_secondary_{source}_{main_branch}"
            start_node = f"junction_main_{source}_{main_branch}"
            end_node = f"junction_second_{source}_{main_branch}"
            diameter = round(random.uniform(0.15, 0.2), 2)
            length = random.randint(500, 800)
            material = 'Steel'
            pipes_data.append([pipe_id, start_node, end_node, diameter, length, material])
            G.add_edge(start_node, end_node, id=pipe_id)

            # Tworzenie trzeciorzędnych rur (third)
            for second_branch in range(1, random.randint(5, 17)):
                pipe_id = f"P_third_{source}_{main_branch}_{second_branch}"
                start_node = f"junction_second_{source}_{main_branch}_{second_branch}"
                end_node = f"junction_third_{source}_{main_branch}_{second_branch}"
                diameter = round(random.uniform(0.1, 0.15), 2)
                length = random.randint(300, 500)
                material = 'PVC'
                pipes_data.append([pipe_id, start_node, end_node, diameter, length, material])
                G.add_edge(start_node, end_node, id=pipe_id)

                # Tworzenie rur do budynków (building)
                for building_branch in range(1, random.randint(4, 25)):
                    pipe_id = f"P_building_{source}_{main_branch}_{second_branch}_{building_branch}"
                    start_node = f"junction_third_{source}_{main_branch}_{second_branch}_{building_branch}"
                    end_node = f"Building_{pipe_counter}"
                    diameter = round(random.uniform(0.08, 0.1), 2)
                    length = random.randint(100, 300)
                    material = 'PVC'
                    pipes_data.append([pipe_id, start_node, end_node, diameter, length, material])
                    G.add_edge(start_node, end_node, id=pipe_id)
                    pipe_counter += 1

                    if pipe_counter > building_target:
                        break
                if pipe_counter > building_target:
                    break
            if pipe_counter > building_target:
                break

        # Dodanie deterministycznych pętli między junction_second, aby stworzyć cykle
        for loop in range(1, 10):
            start_node = f"junction_second_{source}_{loop}"
            end_node = f"junction_second_{source}_{loop + 1 if loop + 1 <= main_branch else 1}"

            # Dodanie połączenia tylko wtedy, gdy nie istnieje jeszcze w sieci
            if start_node != end_node and not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node, id=f"P_loop_{source}_{main_branch}_{loop}")
                cycle = list(nx.cycle_basis(G))
                
                if cycle:
                    pipe_id = f"P_loop_{source}_{main_branch}_{loop}"
                    diameter = round(random.uniform(0.1, 0.15), 2)
                    length = random.randint(400, 600)
                    material = random.choice(['Steel', 'PVC'])
                    pipes_data.append([pipe_id, start_node, end_node, diameter, length, material])
                    print(f"Dodano pętlę: {start_node} -> {end_node}, Pipe_ID: {pipe_id}")
                else:
                    G.remove_edge(start_node, end_node)

        if pipe_counter > building_target:
            break

    # Tworzenie DataFrame
    pipes_df = pd.DataFrame(pipes_data, columns=['Pipe_ID', 'Start_Node', 'End_Node', 'Diameter', 'Length', 'Material'])
    return pipes_df

# Generowanie rur, aby osiągnąć 9171 budynków
pipes_network = generate_pipe_network(9171)

print("Połączenia tworzące potencjalne cykle:")
print(pipes_network[pipes_network['Pipe_ID'].str.contains('P_loop')])

# Zapisz do pliku CSV
pipes_network.to_csv('FlowAlgorithm/pipes_network.csv', index=False)

# Wyświetlenie pierwszych kilku wierszy
print(pipes_network.head())
