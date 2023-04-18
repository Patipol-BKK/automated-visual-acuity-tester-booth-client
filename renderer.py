import pygame

from utils import *
from ui import *

import yaml

config_stream = open("config.yml", 'r')
config = yaml.load(config_stream, Loader=yaml.FullLoader)

DISPLAY_WIDTH = config['display']['width']
DISPLAY_HEIGHT = config['display']['height']

ASPECT_RATIO = config['display']['aspect_ratio'].split('_')
ASPECT_RATIO = [int(x) for x in ASPECT_RATIO]


class Line:
    def __init__(self, figures, height):
        self.figures = figures
        self.height = height
        self.gap = height

        self.figure_offsets = []

        # Calculate total line width
        self.line_width = 0
        for i, figure in enumerate(figures):
            self.figure_offsets.append(
                self.line_width + figure.render(self.height).get_size()[0] / 2)
            self.line_width += figure.render(self.height).get_size()[0]
            self.line_width += self.gap
        self.line_width -= self.gap

    def render(self, display):
        center_x = DISPLAY_WIDTH / 2
        center_y = DISPLAY_HEIGHT / 2

        # Render each figure
        current_x = center_x - self.line_width / 2
        for i, figure in enumerate(self.figures):
            rendered_figure = figure.render(self.height)
            display.blit(rendered_figure,
                         (current_x, center_y - rendered_figure.get_size()[1] / 2))
            current_x += rendered_figure.get_size()[0] + self.gap


class TestScreen:
    def __init__(self, optotype, distance, count, logMAR, display):
        self.optotype = optotype
        self.distance = distance
        self.display = display
        self.count = count
        self.logMAR = logMAR
        figure_size = get_figure_size(
            logMAR,
            self.distance,
            self.optotype.height)
        self.line = Line(self.optotype.choose_random(count), figure_size)

    def render(self, pointed_idx):
        self.line.render(self.display)

        # Arrow
        if pointed_idx >= 0:
            arrow = Arrow()
            arrow_img = arrow.render(get_figure_size(
                self.logMAR * 1.2, self.distance, 5))
            self.display.blit(arrow_img,
                              (DISPLAY_WIDTH / 2 - self.line.line_width / 2 + self.line.figure_offsets[pointed_idx] - arrow_img.get_size()[0] / 2,
                               DISPLAY_HEIGHT / 2 - get_figure_size(self.logMAR * 1.2, self.distance, 8.6)))

        # Render current test info
        str_optotype = 'Optotype: ' + self.optotype.name

        str_figures = 'Displayed Figures: '
        for figure in self.line.figures:
            str_figures += figure.name + ' '

        str_logMAR = 'logMAR: ' + str(self.logMAR)
        str_snellen = 'Snellen: ' + snellen_dict[self.logMAR]

        FONT = 'components/fonts/' + config['font']
        small_font = pygame.font.Font(FONT, 20)

        text = small_font.render(str_optotype, True, (0, 0, 0))
        self.display.blit(text, (10, 10))

        text = small_font.render(str_figures, True, (0, 0, 0))
        self.display.blit(text, (10, 40))

        text = small_font.render(str_logMAR, True, (0, 0, 0))
        self.display.blit(text, (10, 70))

        text = small_font.render(str_snellen, True, (0, 0, 0))
        self.display.blit(text, (10, 100))

        if pointed_idx >= 0:
            str_pointed = 'Pointed Figure: ' + \
                self.line.figures[pointed_idx].name + \
                '   |  ( Index = ' + str(pointed_idx + 1) + ' )'
            text = small_font.render(str_pointed, True, (0, 0, 0))
            self.display.blit(text, (10, 130))

        # Scale Bar
        scale_length = 5
        pygame.draw.line(self.display, (0, 0, 0),
                         (DISPLAY_WIDTH - 40, DISPLAY_HEIGHT - 25),
                         (DISPLAY_WIDTH - 40 - cm_to_pixels(scale_length),
                          DISPLAY_HEIGHT - 25),
                         width=3)
        str_scale = str(scale_length) + ' cm.'
        text = small_font.render(str_scale, True, (0, 0, 0))
        self.display.blit(text, text.get_rect(center=(
            DISPLAY_WIDTH - 40 - cm_to_pixels(scale_length) / 2, DISPLAY_HEIGHT - 45)))

        return self.line.figures
