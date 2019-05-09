## Gridworld

Visualizing dynamic programming and value iteration on a gridworld

## Usage

Calculate the state values for a random policy using dynamic programming, visualize using pygame

```bash
$ python main.py dynamic-programming
```

Find the optimal state values for the grid using value iteration

```bash
$ python main.py value-iteration
```

Calculate the state values for a random policy (no pygame visualization)

```python
#  make a 4x4 grid with the goal state at (1, 1)
grid = GridWorld(4, 4, (1, 1))

#  dynamic programming is on policy, which is a probability distribution over actions
random_policy = {
	action: 0.25 for action in grid.actions
}

dp = DynamicProgramming(
	random_policy, grid
)

state_values = dp.solve()

[[-5.60984837 -4.70220831 -6.26391679 -7.15815431]
 [-4.70220831  0.         -5.47578477 -7.05090145]
 [-6.26391679 -5.47578477 -6.77037966 -7.47804273]
 [-7.15815431 -7.05090145 -7.47804273 -7.79908417]]
```

TODO run the code

## Dependencies

Use python 3.6.5

Main dependencies are `pygame` and `numpy` - test using `pytest`

## pygame on Mojave

```bash
brew install sdl2 sdl2_gfx sdl2_image sdl2_mixer sdl2_net sdl2_ttf

git clone https://github.com/pygame/pygame.git

cd pygame

python setup.py -config -auto -sdl2

python setup.py install
```
