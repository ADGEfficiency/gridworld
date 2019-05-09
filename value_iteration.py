import numpy as np

from dynamic_programming import check_error, bellman
from gridworld import GridWorld


class ValueIteration():

    def __init__(
            self,
            grid
    ):
        self.grid = grid

        self.step = 0

    def forward(self, state_values):
        updated_state_values = [0] * len(self.grid.states)

        for state in self.grid.states:
            all_action_values = self.get_state_action_values(state, state_values)
            updated_state_values[state.idx] = max(all_action_values)

        done, self.step = check_error(self.step, state_values, updated_state_values)

        return updated_state_values, done

    def solve(self):
        state_values = [0] * len(grid.states)
        done = False
        while not done:
            state_values, done = self.forward(state_values)

        print('final value iteration values')
        print(np.array(state_values).reshape(height, width))
        return state_values

    def get_state_values(self, state, state_values):
        """ gets values for all states """
        state_action_values = []

        for next_state, next_state_value in zip(self.grid.states, state_values):
            state_action_values.append(bellman(state, next_state, next_state_value))

        assert len(state_action_values) == len(self.grid.states)
        return state_action_values

    def get_state_action_values(self, state, state_values):
        """ gets values for each action in a specific state """
        action_values = []

        for action in self.grid.actions:
            probs = self.grid.state_actions[state][action]
            state_action_values = self.get_state_values(state, state_values)
            action_values.append(sum([p * b for p, b in zip(probs, state_action_values)]))

        assert len(action_values) == len(self.grid.actions)
        return action_values


if __name__ == '__main__':

    height = 4
    width = 4

    grid = GridWorld(height, width, (1, 1))

    vi = ValueIteration(grid)
    optimal = vi.solve()
