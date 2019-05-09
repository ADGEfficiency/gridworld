from collections import defaultdict, namedtuple

import numpy as np


State = namedtuple('state', ['idx', 'co_ord', 'terminal', 'reward'])
Action = namedtuple('action', ['name', 'delta'])
StateAction = namedtuple('state_action', ['co_ord', 'action'])


class GridWorld():

    actions = [
        Action('up', (-1, 0)),
        Action('down', (1, 0)),
        Action('left', (0, -1)),
        Action('right', (0, 1))
    ]

    def __init__(
            self,
            height=4,
            width=4,
            goal_co_ord=(0, 0),
    ):
        self.height = height
        self.width = width

        assert goal_co_ord[0] < self.height
        assert goal_co_ord[1] < self.width
        self.goal_co_ord = goal_co_ord

        self.states = self.make_state_space()
        self.state_actions = self.make_state_action_space(self.states)

        print(repr(self))

    def __repr__(self):
        return '{}x{} grid - terminal at {}'.format(
            self.height, self.width, self.goal_co_ord)

    def make_state_space(self):
        """ creates a list of all states with their index and co-ordinates """
        states = []
        idx = 0

        for h in range(self.height):
            for w in range(self.width):
                if h == self.goal_co_ord[0] and w == self.goal_co_ord[1]:
                    #  must be zero reward because it is an absorbing state
                    states.append(State(idx, (h, w), True, 0.0))
                else:
                    states.append(State(idx, (h, w), False, -1.0))

                idx += 1

        return states

    def make_state_action_space(self, states):
        state_actions = defaultdict(dict)

        for state in states:
            for action in self.actions:
                next_co_ord = self.add_co_ords(state.co_ord, action.delta)
                next_state = self.find_state_by_co_ord(states, next_co_ord)
                env_probs = self.make_probabilities(state, next_state, states)
                state_actions[state][action] = env_probs

        return state_actions

    def add_co_ords(self, co_ord, delta):
        """ increments a co-ordinate by a delta """
        if co_ord[0] >= self.height:  raise ValueError(
                '{} outside state space'.format(co_ord))

        if co_ord[1] >= self.width:  raise ValueError(
                '{} outside state space'.format(co_ord))

        h, w = [max(0, sum(x)) for x in zip(co_ord, delta)]

        #  keep above zero
        h = max(0, h)
        w = max(0, w)

        #  keep below max
        h = min(h, self.height-1)
        w = min(w, self.width-1)

        return (h, w)

    def find_state_by_co_ord(self, states, co_ord):
        for state in states:
            if state.co_ord == co_ord:
                return state

        raise ValueError('co_ord {} not found in state space')

    def make_probabilities(self, state, next_state, states):
        """ probability distribution over next states """
        probs = [0] * len(states)

        #  if in terminal, stay there
        if state.terminal is True:
            probs[state.idx] = 1.0

        #  else a determinstic transition to next state
        else:
            probs[next_state.idx] = 1.0

        return probs
