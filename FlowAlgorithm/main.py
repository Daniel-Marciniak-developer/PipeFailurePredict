from flow_simulation import FlowSimulation

def main():
    """
    Main function to run the flow simulation.
    """
    simulation = FlowSimulation()
    
    simulation.load_data('PipeFailurePredict--WaterPrime\FlowAlgorithm\data\hourly_demand.csv', 'PipeFailurePredict--WaterPrime\FlowAlgorithm\data\pipes_network.csv')
    simulation.setup_network()
    simulation.simulate_flows(batch_size=4)
    simulation.save_results('flow_results.csv')


if __name__ == '__main__':
    main()
