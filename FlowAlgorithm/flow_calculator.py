import torch
import math

class FlowCalculator:
    """
    Handles flow calculations using the Hardy-Cross method and GPU-accelerated pressure loss calculations.
    """

    WATER_DENSITY = 1000  # kg/m^3
    GRAVITY = 9.81  # m/s^2 Not used yet

    @staticmethod
    def get_pipe_roughness(material: str) -> float:
        """
        Returns the pipe roughness based on the material.
        
        Args:
            material (str): The material of the pipe (e.g., 'Steel', 'PVC').
        
        Returns:
            float: The roughness of the pipe in meters.
        """
        material = material.lower()
        if material == 'steel':
            return 0.000045
        elif material == 'pvc':
            return 0.0000015
        return 0.0001  # default roughness value

    def calculate_pressure_loss_pytorch(self, flows, diameters, lengths, epsilons):
        """
        Calculates pressure loss using PyTorch on GPU (if available).
        
        Args:
            flows (Tensor): Flow rates [num_hours, num_pipes].
            diameters (Tensor): Pipe diameters [num_pipes].
            lengths (Tensor): Pipe lengths [num_pipes].
            epsilons (Tensor): Pipe roughness values [num_pipes].
        
        Returns:
            Tensor: Pressure loss values.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flows = flows.to(device)
        diameters = diameters.to(device)
        lengths = lengths.to(device)
        epsilons = epsilons.to(device)

        velocity = flows / (math.pi * (diameters / 2.0) ** 2)
        viscosity = 1e-6  # m^2/s
        Re = (velocity * diameters) / viscosity

        f = torch.where((Re < 2000.0) & (Re > 0),
                        64.0 / Re,
                        torch.where(Re >= 2000.0, torch.tensor(0.02, device=device), torch.zeros_like(Re)))

        pressure_loss = f * (lengths / diameters) * (FlowCalculator.WATER_DENSITY * velocity ** 2) / 2.0
        return pressure_loss.cpu()

    def hardy_cross_iteration(self, flows, pressure_losses, loops):
        """
        Implements one iteration of the Hardy-Cross method.
        
        Args:
            flows (dict): The current flow rates in the pipes.
            pressure_losses (dict): The pressure losses in the pipes.
            loops (list): List of loops (cycles) in the network.
        
        Returns:
            list: Adjustments made to the flow rates during this iteration.
        """
        adjustments = []
        for loop in loops:
            delta_P = sum(pressure_losses[pipe] * (1.0 if flows[pipe] > 0 else -1.0) for pipe in loop)
            adjustment = -delta_P / len(loop) if len(loop) > 0 else 0.0
            adjustments.append(adjustment)
            for pipe in loop:
                flows[pipe] += adjustment
        return adjustments
