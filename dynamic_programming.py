import numpy as np

from gridworld import GridWorld


def bellman(
        state, next_state, next_state_value, discount=0.9
):
    """ checks for terminal next states """
    if next_state.terminal is False:
        return state.reward + discount * next_state_value
    else:
        return state.reward


def check_error(step, state_values, updated_state_values):
    """ absolute error of value estimates """
    update_error = np.sum(np.array(state_values) -
                          np.array(updated_state_values))

    print('step {} error {}'.format(step, update_error))

    done = False
    threshold = 1.0
    if np.sum(update_error) < threshold:
        done = True
        print('error is less than {} - stopping'.format(threshold))

    step += 1

    return done, step


class DynamicProgramming():
    def __init__(
            self,
            policy,
            grid
    ):
        self.policy = policy
        self.grid = grid
        self.step = 0

    def forward(self, state_values):
        """ single iteration """
        updated_state_values = self.update_state_values(state_values)
        done, self.step = check_error(self.step, state_values, updated_state_values)
        return updated_state_values, done

    def solve(self):
        """ iterate until convergence """
        state_values = [0] * len(self.grid.states)
        done = False
        while not done:
            state_values, done = self.forward(state_values)

        print('final dynamic programming state values')
        print(np.array(state_values).reshape(self.grid.height, self.grid.width))
        return state_values

    def update_state_values(
           self, values
    ):
        """ update value of all states """
        return [
            self.update_state_value(state, values)
            for state in self.grid.states
        ]

    def update_state_value(
            self, state, values
    ):
        """ update value of a single state """
        probs = self.calculate_state_transition_probabilities(
            state
        )

        bellmans = []
        for next_state, next_state_value in zip(self.grid.states, values):
            bellmans.append(bellman(state, next_state, next_state_value))

        return sum([p * b for p, b in zip(probs, bellmans)])

    def calculate_state_transition_probabilities(
            self, state
    ):
        """ calculates the distribution over next states - depends on env and policy """
        state_action = self.grid.state_actions[state]

        state_action_probs = []
        for action, action_prob in self.policy.items():

            state_action_probs.append([action_prob * env_prob
                                  for env_prob in state_action[action]])

        state_action_probs = np.array(state_action_probs).sum(axis=0)
        return state_action_probs


if __name__ == '__main__':

    height = 4
    width = 4

    grid = GridWorld(height, width, (1, 1))
    random_policy = {
        action: 0.25 for action in grid.actions
    }

    dp = DynamicProgramming(
        random_policy, grid
    )
    state_values = dp.solve()
