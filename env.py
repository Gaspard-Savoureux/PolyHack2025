import numpy as np
from agent import Agent


class GridEnv:
    def __init__(self, grid_size=50, num_agent=10, agent_start_pos=(0, 0)):
        """
        A class used to represent the world

        ...

        Attributes
        ----------
        grid_size : int
            represents the width and heigth of the grid. Example: 50 -> 50x50
        num_agent : int
            the name of the animal
        agent_start_pos: (int, int)
            starting position of the agents

        Methods
        -------
        function_name(param=None)
            Description
        """
        self.grid_size = grid_size
        self.vein = []
        self.agents = []
        self.memory = []

        # Initialize agents
        for _ in range(num_agent):
            x, y = np.random.randint(0, grid_size, 2)
            self.agents.append(Agent(x, y))
