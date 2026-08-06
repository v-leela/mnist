import pygame
import numpy as np

GREEN = (0, 234, 0)
WHITE = (255, 255, 255)


class Window:
    def __init__(self, screen, broder_width=7):
        self.screen = screen
        self.screen_size = screen.get_size()
        self.border_width = broder_width
        self.pixel_size = (sorted(self.screen_size)[0] - 2 * broder_width) // 28
        self.draw_height = 28 * self.pixel_size

        # 28x28 image
        self.grid = np.zeros((28, 28), dtype=np.uint8)

    def make_window(self, color=GREEN):
        pygame.draw.line(  # left
            self.screen,
            color,
            (0, 0),
            (0, self.screen_size[1]),
            self.border_width,
        )

        pygame.draw.line(  # down
            self.screen,
            color,
            (0, self.screen_size[1]),
            (self.screen_size[0], self.screen_size[1]),
            self.border_width,
        )

        pygame.draw.line(  # up
            self.screen,
            color,
            (0, 0),
            (self.screen_size[0], 0),
            self.border_width,
        )

        pygame.draw.line(  # right
            self.screen,
            color,
            (self.screen_size[0], 0),
            (self.screen_size[0], self.screen_size[1]),
            self.border_width,
        )

        pygame.draw.line(  # between
            self.screen,
            color,
            (0, 28 * self.pixel_size + self.border_width),
            (self.screen_size[0], 28 * self.pixel_size + self.border_width),
            self.border_width,
        )

    def draw_pixels(self):
        for row in range(28):
            for col in range(28):
                if self.grid[row, col] == 0:
                    continue

                pygame.draw.rect(
                    self.screen,
                    (self.grid[row, col], self.grid[row, col], self.grid[row, col]),
                    (
                        self.border_width + col * self.pixel_size,
                        self.border_width + row * self.pixel_size,
                        self.pixel_size,
                        self.pixel_size,
                    ),
                )

    def draw(self):
        if not pygame.mouse.get_pressed()[0]:
            return

        mx, my = pygame.mouse.get_pos()

        if (
            mx < self.border_width
            or my < self.border_width
            or mx >= self.border_width + 28 * self.pixel_size
            or my >= self.border_width + 28 * self.pixel_size
        ):
            return

        col = (mx - self.border_width) // self.pixel_size
        row = (my - self.border_width) // self.pixel_size

        # 3x3 brush
        BRIGHT = 255
        LIGHT = 255 / 1.2
        FADE = 255 / 2  # 255 / 3

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = row + dr
                c = col + dc

                if not (0 <= r < 28 and 0 <= c < 28):
                    continue

                if dr == 0 and dc == 0:
                    value = BRIGHT
                elif dr in (-1, 1) and dc in (-1, 1):
                    value = FADE
                else:
                    value = LIGHT

                # don't overwrite brighter pixels
                self.grid[r, c] = max(self.grid[r, c], value)

    def clear(self):
        self.grid.fill(0)

    def get_image(self):
        """
        Returns (784,1) image ready for the network
        """

        x = self.grid.astype(np.float32) / 255.0

        return x.reshape(784, 1)

    def show(self):
        self.make_window()

        self.draw()

        self.draw_pixels()
