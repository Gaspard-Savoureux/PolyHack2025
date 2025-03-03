import numpy as np

import random


# from .agent import Agent


class GridEnv:
    def __init__(
        self,
        grid_size=50,
        num_agent=10,
        fov=2,
        agent_start_pos=(0, 0),
        learning_rate=0.9,
        discount_factor=0.99,
        exploration_rate=0.2,
    ):
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
        from video import generate_blobs

        # workaround to load an agent to save file later on
        # thank you python for being such an excellent language 🖕
        self.template_agent = Agent()

        # Initialize agents
        i = 0
        while i < num_agent:

            x, y = np.random.randint(0, grid_size, 2)
            if not ((x, y) in self.agents):
                self.agents[(x, y)] = Agent(
                    fov=fov,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_rate=exploration_rate,
                )
                i += 1
                continue

        # Initialize veins
        self.world = generate_blobs(self.grid_size, self.grid_size, 0.1, 10)
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
        x, y = pos
        valid_x = x >= 0 and x < self.grid_size
        valid_y = y >= 0 and y < self.grid_size
        return not (valid_x and valid_y)

    def occupied(self, pos: (int, int)) -> bool:
        x, y = pos
        another_agent_present = (x, y) in self.agents
        return another_agent_present

    def snapshot(self):
        # self.memory.append(
        #     {
        #         "agents": [pos for pos, agent in self.agents.items()],
        #         "discovered_empty": dict(self.discovered_empty),
        #         "just_discovered_empty": dict(self.just_discovered_empty),
        #         "discovered_vein": dict(self.discovered_vein),
        #         "just_discovered_vein": dict(self.just_discovered_vein),
        #     }
        # )

        self.memory.append(
            {
                "agents": [pos for pos, agent in self.agents.items()],
                "discovered_empty": [pos for pos, _ in self.discovered_empty.items()],
                "just_discovered_empty": [
                    pos for pos, _ in self.just_discovered_empty.items()
                ],
                "discovered_vein": [pos for pos, _ in self.discovered_vein.items()],
                "just_discovered_vein": [
                    pos for pos, _ in self.just_discovered_vein.items()
                ],
            }
        )

    # backup function
    def render(self):
        fig, ax = plt.subplots()
        grid = np.zeros((self.grid_size, self.grid_size, 3))
        im = ax.imshow(grid, interpolation="nearest")

        def update(frame):
            grid = np.zeros((self.grid_size, self.grid_size, 3))
            grid[:, :, 1] = self.vegetation * 0.5
            for x, y, agent_type in self.memory[frame]:
                if agent_type == 1:
                    grid[x, y] = [0, 0, 1]
                else:
                    grid[x, y] = [1, 0, 0]
            im.set_data(grid)
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.memory), interval=200, blit=True
        )
        plt.title("Predator-Prey Simulation")
        plt.axis("off")
        ani.save("simulation.mp4", writer="ffmpeg", fps=5)
        plt.close()

    #
    def apply_action(self, position, agent, action):
        x, y = position

        # Déterminer la nouvelle position en fonction de l'action
        if action == 0:  # Up
            new_position = (x, y - 1)
        elif action == 1:  # Down
            new_position = (x, y + 1)
        elif action == 2:  # Left
            new_position = (x - 1, y)
        elif action == 3:  # Right
            new_position = (x + 1, y)
        else:
            new_position = position  # Pas de mouvement si action invalide

        # Vérifier si la position est valide
        return new_position if self.valid_pos(new_position) else position

    def step(self):
        self.snapshot()
        new_agent_positions = {}  # Dictionnaire pour stocker les nouvelles positions

        for position, agent in list(self.agents.items()):
            state = agent.state
            action = agent.choose_action(state, self)

            next_position = self.apply_action(position, agent, action)

            next_state = agent.get_state(self, next_position)
            agent.state = next_state
            reward = next_state.get_reward()

            # Update q_table
            agent.update_q_table(state, action, reward, next_state)

            # Ajouter l'agent à sa nouvelle position seulement si elle est libre
            if next_position not in new_agent_positions:
                new_agent_positions[next_position] = agent
            else:
                new_agent_positions[position] = (
                    agent  # Garde l'agent à sa position initiale
                )

        # Mettre à jour la grille des agents
        self.agents = new_agent_positions

    def train(self, num_steps=50, filename="agent.pkl"):
        # print("nb agents: ", len(self.agents))

        try:
            self.template_agent.load_q_table(filename)
        except FileNotFoundError:
            pass

        for step in range(num_steps):
            self.step()

            # world_copy = np.copy(self.world)

            # for x, y in self.discovered_vein.keys():
            #     world_copy[x][y] = 3
            # for x, y in self.agents.keys():
            #     world_copy[x][y] = 2

            # Debug
            # print(world_copy)

        self.template_agent.save_q_table(filename)
        self.snapshot()
        # Debug
        # print("nb agents: ", len(self.agents))
        # print(self.memory)

    def simulate(self, num_steps: int = 100, filename="agent.pkl"):
        try:
            self.template_agent.load_q_table(filename)
        except FileNotFoundError:
            pass

        for step in range(num_steps):
            self.step()

            world_copy = np.copy(self.world)

            for x, y in self.discovered_vein.keys():
                world_copy[x][y] = 3
            for x, y in self.agents.keys():
                world_copy[x][y] = 2

            # Debug
            # print(world_copy)

        self.snapshot()
        # Debug
        # print("nb agents: ", len(self.agents))
        # print(self.memory)
