""" tests of individual functions in gridworld """
import sys
sys.path.append('..')

import pytest

from gridworld import *


@pytest.fixture(scope='function')
def states():
    return [
        State(0, (0, 0), reward=1, terminal=False),
        State(1, (0, 1), reward=-1, terminal=False),
        State(1, (1, 0), reward=-1, terminal=False),
        State(2, (1, 1), reward=0, terminal=True)
    ]


@pytest.fixture(scope='function')
def grid():
    return GridWorld(3, 4)


@pytest.mark.parametrize(
    'height, width',
    [(3, 2), (10, 10), (1, 5)]
)
def test_make_state_space(height, width):
    """ checks state space dimensionality and unique indexes """

    grid = GridWorld(height, width)
    states = grid.states

    assert len(states) == height * width

    idxs = [state.idx for state in states]
    assert len(set(idxs)) == len(idxs)


@pytest.mark.parametrize(
    'height, width, co_ord, delta, expected',
    [
        (4, 4, (0, 0), (-1, 0), (0, 0)),
        (4, 4, (3, 3), (-1, 0), (2, 3)),
        (3, 3, (1, 0), (0, -1), (1, 0)),
        (1, 3, (0, 1), (0, -1), (0, 0)),
        (1, 3, (0, 1), (0, 1), (0, 2))
    ]
)
def test_add_cords(height, width, co_ord, delta, expected):
    """ checks we add deltas to co-oridnates correctly """
    grid = GridWorld(height, width)
    new_co_ord = grid.add_co_ords(co_ord, delta)
    assert new_co_ord == expected


def test_find_state_by_co_ord(grid, states):
    """ checks we find the correct states by co-ordinate """

    for state in states:
        co_ord = state.co_ord
        found_state = grid.find_state_by_co_ord(states, co_ord)

    assert found_state == state

@pytest.mark.parametrize(
    'state, next_state',
    [
        (State(0, (0, 0), reward=1, terminal=False),
         State(1, (0, 1), reward=-1, terminal=False))
    ]
)
def test_make_probabilities(grid, states, state, next_state):
    """ checks we form the probability distribution over states correctly """

    probs = grid.make_probabilities(state, next_state, states)

    if state.terminal is False:
        assert probs[next_state.idx] == 1.0
    elif state.terminal is True:
        assert probs[state.idx] == 1.0

    assert sum(probs) == 1.0


def test_make_state_action_space(grid):
    """ checks dimensionality of state-action space """

    height = 2
    width = 2

    grid = GridWorld(height, width)
    state_actions = grid.state_actions

    dim = 0
    for state, actions in state_actions.items():
        for action in actions:
            dim += 1

    assert dim == height * width * len(actions.keys())




