import numpy as np
from env import GridEnv
from enum import Enum


class State:
    """
    The state of an agent at a given time
    """

    def __init__(self, grid):
        self.grid = grid

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


class Agent:
    """
    A class used to represent the world

    ...

    Attributes
    ----------
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
    Methods
    -------
    function_name(param=None)
        Description
    """

    def __init__(
        self,
        x: int,
        y: int,
        fov: int = 3,
        learning_rate: float = 0.90,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.2,
    ):
        self.x = x
        self.y = y
        self.fov = fov
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        # self.actions = [] // Possibly useless, will use class Action

    def choose_action(self, state, env):
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

    # def update_q_table(self, state, action, reward, next_state):

    # def save_q_table(cls, filename):

    # def load_q_table(cls, filename):
