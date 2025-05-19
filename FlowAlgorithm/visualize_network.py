# visualize_network.py

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import time
import pickle

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Funkcja ustawia pozycje węzłów w układzie hierarchicznym (drzewa genealogicznego).

    Args:
        G: Graf skierowany (DiGraph).
        root: Węzeł korzenia. Jeśli None, wybrany zostanie pierwszy węzeł z listy.
        width: Rozmiar układu w osi X.
        vert_gap: Odstęp między poziomami w osi Y.
        vert_loc: Lokalizacja korzenia w osi Y.
        xcenter: Położenie korzenia w osi X.

    Returns:
        Słownik z pozycjami węzłów.
    """
    if not nx.is_tree(G):
        print("Graf nie jest drzewem. Próba zastosowania hierarchicznego układu może nie działać poprawnie.")

    if root is None:
        root = next(iter(nx.topological_sort(G)))  # Wybierz pierwszy węzeł w topologicznym sortowaniu

    def _hierarchy_pos(G, root, left, right, vert_gap, vert_loc, pos=None, parent=None):
        if pos is None:
            pos = {}
        pos[root] = ((left + right) / 2, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = (right - left) / len(children)
            next_left = left
            for child in children:
                next_right = next_left + dx
                pos = _hierarchy_pos(G, child, next_left, next_right, vert_gap, vert_loc - vert_gap, pos, root)
                next_left += dx
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc)

class NetworkVisualizer:
    """
    Klasa odpowiedzialna za wizualizację topologii sieci hydraulicznej,
    uwzględniającą tylko główne rury i budynki z zapotrzebowaniem na pierwszą godzinę.
    """

    def __init__(self, data_dir: str, output_dir: str):
        """
        Inicjalizuje wizualizator z odpowiednimi ścieżkami do danych i wyjść.

        Args:
            data_dir (str): Ścieżka do katalogu z danymi CSV.
            output_dir (str): Ścieżka do katalogu, gdzie będą zapisywane wyniki wizualizacji.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.main_flows_df = None
        self.pipes_df = None
        self.nodes_df = None
        self.demands_df = None
        self.G = nx.DiGraph()

    def load_data(self):
        """
        Wczytuje dane z plików CSV do DataFrame.
        """
        print("Rozpoczynanie ładowania danych...")
        start_time = time.time()
        try:
            self.main_flows_df = pd.read_csv(os.path.join(self.data_dir, 'main_flows.csv'))
            print("Załadowano 'main_flows.csv'")
            self.pipes_df = pd.read_csv(os.path.join(self.data_dir, 'pipes_network.csv'))
            print("Załadowano 'pipes_network.csv'")
            self.nodes_df = pd.read_csv(os.path.join(self.data_dir, 'node_network.csv'))
            print("Załadowano 'node_network.csv'")
            self.demands_df = pd.read_csv(os.path.join(self.data_dir, 'hourly_demand.csv'))
            print("Załadowano 'hourly_demand.csv'")
            print("Dane zostały pomyślnie wczytane.")
        except FileNotFoundError as e:
            print(f"Nie znaleziono pliku: {e.filename}")
            raise
        except pd.errors.ParserError as e:
            print(f"Błąd parsowania pliku CSV: {e}")
            raise
        end_time = time.time()
        print(f"Ładowanie danych zajęło {end_time - start_time:.2f} sekund.\n")

    def build_graph(self):
        """
        Buduje graf skierowany na podstawie danych o głównych rurach i budynkach.
        """
        print("Rozpoczynanie budowy grafu...")
        start_time = time.time()
        if self.main_flows_df is None or self.pipes_df is None or self.nodes_df is None or self.demands_df is None:
            raise ValueError("Dane nie zostały wczytane. Wywołaj metodę load_data() przed build_graph().")

        # Wybierz tylko główne rury na podstawie Pipe_ID z main_flows.csv
        main_pipes = self.main_flows_df['Pipe_ID'].tolist()
        main_pipes_df = self.pipes_df[self.pipes_df['Pipe_ID'].isin(main_pipes)]
        print(f"Wybrano {len(main_pipes)} głównych rur z 'main_flows.csv'.")

        # Dodaj węzły z atrybutami
        print("Dodawanie węzłów do grafu...")
        nodes = self.nodes_df.to_dict('records')
        building_demands = self.demands_df.set_index('Building')['Hour_0'].to_dict()
        for idx, row in enumerate(nodes, start=1):
            node_id = row['Node_ID']
            node_type = row['Node_Type']
            demand = building_demands.get(node_id, 0.0) if node_type == 'Building' else 0.0
            self.G.add_node(node_id, type=node_type, demand=demand)
            if idx % 1000 == 0 or idx == len(nodes):
                print(f"Dodano {idx}/{len(nodes)} węzłów.")

        # Dodaj krawędzie (rury) z atrybutami
        print("Dodawanie krawędzi (rur) do grafu...")
        edges = main_pipes_df.to_dict('records')
        for idx, row in enumerate(edges, start=1):
            pipe_id = row['Pipe_ID']
            start = row['Start_Node']
            end = row['End_Node']
            diameter = row['Diameter']
            length = row['Length']
            material = row['Material']
            # Pobierz przepływ dla pierwszej godziny z main_flows.csv
            flow = self.main_flows_df[self.main_flows_df['Pipe_ID'] == pipe_id]['Hour_0'].values
            flow = flow[0] if len(flow) > 0 else 0.0
            self.G.add_edge(start, end, Pipe_ID=pipe_id, Diameter=diameter, Length=length, Material=material, Flow=flow)
            if idx % 1000 == 0 or idx == len(edges):
                print(f"Dodano {idx}/{len(edges)} krawędzi.")

        end_time = time.time()
        print(f"Budowa grafu zajęła {end_time - start_time:.2f} sekund.")
        print("Graf został pomyślnie zbudowany.\n")

    def visualize(self, root='Source_1'):
        """
        Wizualizuje graf sieci hydraulicznej jako drzewo genealogiczne,
        uwzględniając tylko główne rury i budynki z zapotrzebowaniem na pierwszą godzinę.

        Args:
            root (str): Węzeł korzenia (domyślnie 'Source_1').
        """
        print("Rozpoczynanie wizualizacji grafu...")
        start_time = time.time()
        try:
            # Sprawdź, czy root istnieje
            if root not in self.G:
                print(f"Węzeł korzenia '{root}' nie istnieje w grafie.")
                return

            # Tworzenie podgrafu osiągalnego z root
            sub_nodes = nx.descendants(self.G, root)
            sub_nodes.add(root)
            subgraph = self.G.subgraph(sub_nodes).copy()
            print(f"Utworzono podgraf z {len(subgraph.nodes())} węzłów i {len(subgraph.edges())} krawędzi.")

            plt.figure(figsize=(20, 15))
            print("Tworzenie układu grafu (hierarchiczny)...")
            pos = hierarchy_pos(subgraph, root=root)
            print("Układ grafu został stworzony.")

            # Rysuj węzły z kolorami zależnymi od typu
            print("Rysowanie węzłów...")
            node_colors = ['green' if data['type'] == 'Source' else
                           'orange' if data['type'] == 'Junction' else
                           'blue' if data['type'] == 'Building' else
                           'grey' for node, data in subgraph.nodes(data=True)]
            nx.draw_networkx_nodes(subgraph, pos, node_size=700, node_color=node_colors, alpha=0.9)
            print("Węzły zostały narysowane.")

            # Rysuj krawędzie z jednolitym kolorem
            print("Rysowanie krawędzi...")
            nx.draw_networkx_edges(subgraph, pos, edge_color='black', width=1.0, arrows=True, arrowstyle='-|>', arrowsize=15)
            print("Krawędzie zostały narysowane.")

            # Rysuj etykiety węzłów
            print("Dodawanie etykiet do węzłów...")
            labels = {node: f"{node}\nDemand: {data['demand']} m³/h" if data['type'] == 'Building' else node
                      for node, data in subgraph.nodes(data=True)}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
            print("Etykiety węzłów zostały dodane.")

            # Rysuj etykiety krawędzi (Pipe_ID)
            print("Dodawanie etykiet do krawędzi...")
            edge_labels = { (u, v): f"{data['Pipe_ID']}" 
                           for u, v, data in subgraph.edges(data=True)}
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='black', font_size=6)
            print("Etykiety krawędzi zostały dodane.")

            # Dodaj legendę
            print("Dodawanie legendy...")
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', edgecolor='green', label='Source'),
                Patch(facecolor='orange', edgecolor='orange', label='Junction'),
                Patch(facecolor='blue', edgecolor='blue', label='Building'),
                Patch(facecolor='grey', edgecolor='grey', label='Other')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            print("Legenda została dodana.")

            # Finalizacja wykresu
            plt.title(f"Topologia Sieci Hydraulicznej - Drzewo od '{root}' (Godzina 0)", fontsize=20)
            plt.axis('off')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'network_topology_{root}_hour0.png')
            print(f"Zapisywanie wizualizacji do pliku '{output_path}'...")
            plt.savefig(output_path, dpi=300)
            print("Wizualizacja została zapisana.")
            plt.show()
            print("Wizualizacja została wyświetlona.\n")
        except Exception as e:
            print(f"Wystąpił błąd podczas wizualizacji: {e}")
            raise
        end_time = time.time()
        print(f"Wizualizacja zajęła {end_time - start_time:.2f} sekund.\n")

def main():
    # Definicja ścieżek do danych i wyjść
    project_dir = 'PipeFailurePredict--WaterPrime'
    flow_algorithm_dir = os.path.join(project_dir, 'FlowAlgorithm')
    data_directory = os.path.join(flow_algorithm_dir, 'data')
    output_directory = os.path.join(flow_algorithm_dir, 'output')

    # Upewnij się, że katalog wyjściowy istnieje
    os.makedirs(output_directory, exist_ok=True)

    # Inicjalizacja wizualizatora
    visualizer = NetworkVisualizer(data_dir=data_directory, output_dir=output_directory)

    # Krok 1: Wczytaj dane
    visualizer.load_data()

    # Krok 2: Zbuduj graf
    visualizer.build_graph()

    # Krok 3: Wizualizacja grafu tylko od 'Source_1'
    visualizer.visualize(root='Source_1')

if __name__ == "__main__":
    main()
