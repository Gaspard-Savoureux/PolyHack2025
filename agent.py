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
        # x: int,
        # y: int,
        fov: int = 3,
        learning_rate: float = 0.90,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.2,
    ):
        # self.x = x
        # self.y = y
        self.fov = fov
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.state = State([0])  # CHARGER L'etat reel
        # self.actions = [] // Possibly useless, will use class Action

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

    # TODO maybe rename get_sensor_output (for roleplaying reasons)
    def get_state(self, env: GridEnv, pos: (int, int)):
        x, y = pos
        fov = self.fov
        state_grid = []
        for dx in range(-fov, fov + 1):
            for dy in range(-fov, fov + 1):
                new_x = x + dx
                new_y = y + dy

                # if invalid pos, we skip
                if not env.valid_pos((new_x, new_y)):
                    continue

                # Get value of the pos
                # pos = env.world[new_x][new_y]
                state_grid.append(env.world[new_x][new_y])

                # We add the newly discovered vein
                # print("pos: ", pos)
                # env.discovered_vein[pos] = True

        return State(np.array(state_grid))

    def step(self, pos: (int, int), env: GridEnv, action: int) -> ((int, int), float):
        """ """
        # print("pos: ", pos)
        new_x, new_y = pos

        match action:
            case Action.UP:
                new_y -= 1
            case Action.DOWN:
                new_y += 1
            case Action.LEFT:
                new_x -= 1
            case Action.RIGHT:
                new_x += 1

        reward = -1

        # Agent cannot physically go to the next cell
        if not env.valid_pos((new_x, new_y)):
            return ((pos[0], pos[1]), reward - 10)

        cell = env.world[new_x][new_y]

        match cell:
            case 0:  # next cell is empty
                return ((new_x, new_y), reward)
            case 1:  # next cell is part of vein
                return ((pos[0], pos[1]), reward + 10)
