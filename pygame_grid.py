import pygame

from dynamic_programming import DynamicProgramming
from gridworld import GridWorld
from value_iteration import ValueIteration


class Tile():
    border_width = 2
    border_color = pygame.Color('gray')

    def __init__(
            self, surface, height, width, goal_co_ord, tile_size
    ):
        self.surface = surface
        self.size = tile_size

        self.co_ord = (height, width)
        self.position = self.convert_co_ord_to_position(self.co_ord)

        self.goal_position = self.convert_co_ord_to_position(goal_co_ord)

    def __repr__(self):
        return 'pygame tile at {}'.format(self.co_ord)

    def convert_co_ord_to_position(self, co_ord):
        return (co_ord[0] * self.size[0],
                co_ord[1] * self.size[1])

    def draw(self, state_value, state_values):

        rect = pygame.Rect(self.position, self.size)

        #  ordinary tile
        # pygame.draw.rect(self.surface, pygame.Color('white'), rect, 0)

        minimum = min(state_values) + 0.0001
        normed_state_value = (state_value - minimum) / (max(state_values) -
                                                        minimum)
        pygame.draw.rect(self.surface, (0, 80 + 100 * normed_state_value, 0, 0), rect, 0)

        #  terminal aka goal tile
        if self.goal_position == self.position:
            pygame.draw.rect(self.surface, (0, 180, 0), rect, 0)

        #  drawing state value
        basicfont = pygame.font.SysFont(None, 48)
        text = basicfont.render(
            '{:2.1f}'.format(state_value), True, pygame.Color('white')
        )

        textrect = text.get_rect()
        textrect.centerx = self.position[0] + self.size[0] / 2
        textrect.centery = self.position[1] + self.size[1] / 2

        self.surface.blit(text, textrect)

        #  drawing border color
        pygame.draw.rect(self.surface, self.border_color, rect, self.border_width)


def make_pygame(height, width, goal_co_ord, tile_size):
    """ makes pygame surface and tiles """
    surface_size = (height * tile_size[0], width * tile_size[1])
    surface = pygame.display.set_mode(surface_size)
    surface.fill(pygame.Color('black'))

    tiles = []
    for h in range(height):
        for w in range(width):
            tiles.append(Tile(surface, h, w, goal_co_ord, tile_size))
    return surface, tiles


def update_tiles(state_values, tiles):
    [tile.draw(value, state_values) for value, tile in zip(state_values, tiles)]


def check_pygame():
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            pygame.quit()
            return False
    return True


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--solver', default='dynamic-programming', nargs='?')
    # args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption('grid')

    height = 8
    width = 8
    goal_co_ord = (3, 2)
    tile_size = (200, 200)

    surface, tiles = make_pygame(height, width, goal_co_ord, tile_size)

    grid = GridWorld(height, width, goal_co_ord=goal_co_ord)

    state_values = [0] * len(grid.states)
    update_tiles(state_values, tiles)

    random_policy = {
        action: 0.25 for action in grid.actions
    }

    dp = DynamicProgramming(random_policy, grid)
    vi = ValueIteration(grid)

    running = True
    done = False
    # while not done:
    while running:
        running = check_pygame()
        pygame.display.update()
        state_values, done = dp.forward(state_values)
        update_tiles(state_values, tiles)
