import sys
sys.path.append('..')

import numpy as np
import pytest

from dynamic_programming import bellman
from gridworld import GridWorld, State


@pytest.fixture(scope='function')
def states():
    return [
        State(0, (0, 0), reward=1, terminal=False),
        State(1, (0, 1), reward=-1, terminal=False),
        State(1, (1, 0), reward=-1, terminal=False),
        State(2, (1, 1), reward=0, terminal=True)
    ]

@pytest.fixture(scope='function')
def next_states():
    return [
        State(1, (0, 1), reward=-1, terminal=False),
        State(0, (0, 0), reward=1, terminal=False),
        State(2, (1, 1), reward=0, terminal=True),
        State(2, (1, 1), reward=0, terminal=True)
    ]

def test_bellman_equation(states, next_states):
    next_state_values = [10, -5, 12, 100]

    bellmans = []

    for state, next_state, next_state_value in zip(states, next_states, next_state_values):
        bellmans.append(bellman(state, next_state, next_state_value))

    expected_value = [
        1 + 0.9 * 10, -1 + 0.9 * -5, -1, 0
    ]

    np.testing.assert_array_almost_equal(bellmans, expected_value)
