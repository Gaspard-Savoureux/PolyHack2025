# Python librairies
import numpy as np
import random
import pickle

from collections import defaultdict

# local files
from environment import GridEnv


class State:
    """
    The state of an agent at a given time
    """

    def __init__(self, grid):
        self.grid = grid
        self.reward = self.get_reward()

    def get_reward(self):
        reward_mapping = {
            0: -10,  # WALL
            1: -1,  # OTHER_AGENT
            2: -1,  # DISCOVERED_EMPTY
            3: -1,  # DISCOVERED_MINERAL
            4: 4,  # JUST_DISCOVERED_EMPTY
            5: 25,  # JUST_DISCOVERED_MINERAL
        }
        return sum(reward_mapping.get(cell, 0) for cell in self.grid)

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    def __hash__(self):
        return hash(tuple(self.grid))

    def get_key(self):
        return tuple(self.grid)


class Action:
    """
    Simple class that will act as an enum
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType:
    WALL = 0
    OTHER_AGENT = 1
    DISCOVERED_EMPTY = 2
    DISCOVERED_MINERAL = 3
    JUST_DISCOVERED_EMPTY = 4
    JUST_DISCOVERED_MINERAL = 5
    SELF = 6


class Agent:
    """
    Class that represent agents that will work as a swarm.
    ...

    Attributes
    ----------
    q_table: defaultdict(float)
      common q_table for the agent

    Methods
    -------
    __init__(
      x : int
          initial x coordinate
      y : int
          initial y coordinate
      fov: int
        field of view of the agent
      # The following are self explatory
      learning_rate: float,
      discount_factor: float,
      exploration_rate: float,
    )
        Create an agent
    """

    q_table = defaultdict(float)

    def __init__(
        self,
        fov: int = 3,
        learning_rate: float = 0.90,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.2,
    ):
        self.fov = fov
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.state = State([0])  # CHARGER L'etat reel

    def choose_action(self, state, env) -> int:
        actions = [
            value for key, value in vars(Action).items() if not key.startswith("__")
        ]

        if np.random.random() < self.exploration_rate:  # Exloration
            return np.random.choice(actions)
        else:  # Best action
            state_key = state.get_key()
            q_values = {a: self.__class__.q_table[(state_key, a)] for a in actions}
            max_q = max(q_values.values()) if q_values else 0

            # All actions with equal max value
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions) if best_actions else None

    def update_q_table(self, state, action, reward, next_state):
        state_key = state.get_key()
        next_key = next_state.get_key() if next_state else None
        current_q = self.__class__.q_table.get((state_key, action), 0)

        actions = [
            value for key, value in vars(Action).items() if not key.startswith("__")
        ]

        next_max = (
            max([self.__class__.q_table.get((next_key, a), 0) for a in actions])
            if next_key
            else 0
        )

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max - current_q
        )

        self.__class__.q_table[(state_key, action)] = new_q

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            self.q_table.update(pickle.load(f))

    # Shamefully unreadable code
    def get_state(self, env: GridEnv, pos: (int, int)):
        """
        0: wall
        1: other_robot
        2: discovered_empty
        3: discovered_mineral
        4: just_discovered_empty
        5: just_discovered_mineral
        """
        x, y = pos
        fov = self.fov
        state_grid = []
        for dx in range(-fov, fov + 1):
            for dy in range(-fov, fov + 1):
                new_x = x + dx
                new_y = y + dy

                if env.out_of_bound((new_x, new_y)):  # out_of_bound
                    state_grid.append(CellType.WALL)

                elif env.occupied((new_x, new_y)) and (
                    new_x != x and new_y != y
                ):  # occupied by another agent
                    state_grid.append(CellType.OTHER_AGENT)
                elif (new_x, new_y) in env.just_discovered_empty:
                    del env.just_discovered_empty[(new_x, new_y)]
                    env.discovered_empty[(new_x, new_y)] = 1
                    state_grid.append(CellType.DISCOVERED_EMPTY)
                elif (
                    new_x,
                    new_y,
                ) in env.just_discovered_vein:  # next pos is part of vein
                    del env.just_discovered_vein[(new_x, new_y)]
                    env.discovered_vein[(new_x, new_y)] = 1
                    state_grid.append(CellType.DISCOVERED_MINERAL)
                elif (new_x, new_y) in env.discovered_empty:  # next post  is empty
                    state_grid.append(CellType.DISCOVERED_EMPTY)
                elif (new_x, new_y) in env.discovered_vein:
                    state_grid.append(CellType.DISCOVERED_MINERAL)
                elif env.world[new_x][new_y] == 1:
                    env.just_discovered_vein[(new_x, new_y)] = 1
                    state_grid.append(CellType.JUST_DISCOVERED_MINERAL)
                else:
                    env.just_discovered_empty[(new_x, new_y)] = 1
                    state_grid.append(CellType.JUST_DISCOVERED_EMPTY)

        return State(np.array(state_grid))
