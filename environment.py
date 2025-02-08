import numpy as np

import random

# from .agent import Agent


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
        # Array of size : grid-size
        # if (x,y) = 1 -> mineral
        # else -> empty
        self.world = np.array([[0] * grid_size] * grid_size)
        # self.world = [[0] * grid_size] * grid_size
        # we need a representation that differentiates between recently discovered minerals
        # and minerals that were discovered previously
        # self.recently_discovered = []
        self.discovered_empty = {}
        self.just_discovered_empty = {}
        self.discovered_vein = {}
        self.just_discovered_vein = {}
        self.agents = {}  # {(k: pos, v: agent)}
        self.memory = []
        # Delay import to avoid circular dependency
        from agent import Agent

        # Initialize agents
        i = 0
        while i < num_agent:

            x, y = np.random.randint(0, grid_size, 2)
            if not ((x, y) in self.agents):
                self.agents[(x, y)] = Agent(fov=3)
                i += 1
                continue

        # Initialize veins
        for i in range(grid_size - 1, grid_size - 4, -1):
            for j in range(grid_size - 1, grid_size - 4, -1):
                self.world[i][j] = 1

    def valid_pos(self, pos: (int, int)) -> bool:
        x, y = pos
        valid_x = x >= 0 and x < self.grid_size
        valid_y = y >= 0 and y < self.grid_size
        another_agent_present = (x, y) in self.agents
        return valid_x and valid_y and not another_agent_present

    def out_of_bound(self, pos: (int, int)) -> bool:
        valid_x = x >= 0 and x < self.grid_size
        valid_y = y >= 0 and y < self.grid_size
        return valid_x and valid_y

    def occupied(self, pos: (int, int)) -> bool:
        another_agent_present = (x, y) in self.agents
        return another_agent_present

    def step(self):
        for position, agent in list(self.agents.items()):
            state = agent.state
            action = agent.choose_action(state, self)

            next_state = agent.get_state(self, position)

            (next_position, reward) = agent.step(position, self, action)

            # Update q_table
            agent.update_q_table(state, action, reward, next_state)

            if position != next_position:
                # self.agents[next_position].append(self.agents.pop(i))
                self.agents[next_position] = self.agents.pop(position)
            else:
                agent.state = next_state
                # self.agents[next_position].state = next_state

    def train(self, num_steps=50):
        print("nb agents: ", len(self.agents))
        for step in range(num_steps):
            self.step()

            # Debug
            # agents_pos = [pos for pos, agent in list(self.agents.items())]
            # for pos, agent in list(self.agents.items()):
            #     print("pos", pos, " agent", agent)
            # print("nb agents: ", len(self.agents))
            world_copy = np.copy(self.world)

            for x, y in self.discovered_vein.keys():
                world_copy[x][y] = 3
            for x, y in self.agents.keys():
                world_copy[x][y] = 2

            print(world_copy)

        print("nb agents: ", len(self.agents))
