import networkx as nx
import pandas as pd

class HydraulicNetwork:
    """
    Manages the hydraulic network represented as a directed graph.
    Responsible for adding nodes, pipes, and finding loops within the network.
    """

    def __init__(self):
        self.network = nx.DiGraph()

    def add_node(self, node_id: str, node_type: str, demand: float = 0.0):
        """
        Adds a node to the network.
        
        Args:
            node_id (str): The identifier of the node.
            node_type (str): The type of the node (e.g., 'Building', 'Source').
            demand (float): The demand associated with the node (default: 0.0).
        """
        self.network.add_node(node_id, type=node_type, demand=demand)

    def add_pipe(self, start_node: str, end_node: str, pipe_id: str, diameter: float, length: float, material: str):
        """
        Adds a pipe (edge) between nodes with additional attributes.
        
        Args:
            start_node (str): The starting node of the pipe.
            end_node (str): The ending node of the pipe.
            pipe_id (str): The unique identifier of the pipe.
            diameter (float): Diameter of the pipe.
            length (float): Length of the pipe.
            material (str): Material of the pipe.
        """
        self.network.add_edge(start_node, end_node, id=pipe_id, diameter=diameter, length=length, material=material, flow=0.0)

    def find_loops(self):
        """
        Finds all loops (cycles) in the network and maps them to pipe IDs.
        
        Returns:
            list: A list of lists where each inner list contains pipe IDs forming a loop.
        """
        undirected_graph = self.network.to_undirected()
        node_cycles = nx.cycle_basis(undirected_graph)
        pipe_cycles = []

        for cycle in node_cycles:
            pipes = []
            num_nodes = len(cycle)
            for i in range(num_nodes):
                u = cycle[i]
                v = cycle[(i + 1) % num_nodes]
                pipe_id = self._get_pipe_id_between_nodes(u, v)
                if pipe_id:
                    pipes.append(pipe_id)
            if pipes:
                pipe_cycles.append(pipes)
        return pipe_cycles

    def _get_pipe_id_between_nodes(self, u, v):
        """
        Helper function to get pipe ID between two nodes.
        
        Args:
            u (str): First node ID.
            v (str): Second node ID.
        
        Returns:
            str: Pipe ID if exists, else None.
        """
        if self.network.has_edge(u, v):
            return self.network[u][v]['id']
        elif self.network.has_edge(v, u):
            return self.network[v][u]['id']
        return None
