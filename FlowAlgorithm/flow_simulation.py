from flow_calculator import FlowCalculator
from hydraulic_network import HydraulicNetwork
from data_loader import DataLoader
import os


class FlowSimulation:
    """
    Coordinates the flow simulation by preparing the network, calculating flows, and managing batch processing.
    """

    def __init__(self):
        self.network = HydraulicNetwork()
        self.flow_calculator = FlowCalculator()
        self.demands = None
        self.pipes = None

    def load_data(self, demand_file: str, pipe_file: str):
        """
        Loads the demand and pipe data.
        
        Args:
            demand_file (str): Path to the demand data CSV.
            pipe_file (str): Path to the pipe data CSV.
        """
        self.demands = DataLoader.load_demands(demand_file)
        self.pipes = DataLoader.load_pipes(pipe_file)

    def setup_network(self):
        """
        Sets up the hydraulic network by adding nodes and pipes.
        Calculates an average demand per building if only a single demand value is needed.
        """
        self.demands['Demand'] = self.demands.loc[:, 'Hour_0':'Hour_23'].mean(axis=1)

        for _, row in self.pipes.iterrows():
            self.network.add_pipe(row['Start_Node'], row['End_Node'], row['Pipe_ID'], row['Diameter'], row['Length'], row['Material'])

        for node in self.demands['Building']:
            demand = self.demands.loc[self.demands['Building'] == node, 'Demand'].values[0]
            self.network.add_node(node, 'Building', demand=demand)


    def simulate_flows(self, batch_size=4):
        """
        Runs the flow simulation in batches.
        
        Args:
            batch_size (int): The number of hours to process in each batch.
        """
        pass

    def save_results(self, file_name: str):
        """
        Save the simulation results into the output folder.
        
        Args:
            file_name (str): The name of the output file.
        """
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, file_name)
        print(f"Results saved to {file_path}")
