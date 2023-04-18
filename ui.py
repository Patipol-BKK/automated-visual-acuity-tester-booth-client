import math
import io
import pygame
import yaml

from utils import *

config_stream = open("config.yml", 'r')
config = yaml.load(config_stream, Loader=yaml.FullLoader)

DISPLAY_WIDTH = config['display']['width']
DISPLAY_HEIGHT = config['display']['height']

ASPECT_RATIO = config['display']['aspect_ratio'].split('_')
ASPECT_RATIO = [int(x) for x in ASPECT_RATIO]

# Display Dimensions in cm
DISPLAY_DIAG_DIM = config['display']['screen_size'] * 2.54
DISPLAY_WIDTH_DIM = ASPECT_RATIO[0] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))
DISPLAY_HEIGHT_DIM = ASPECT_RATIO[1] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))


class Arrow:
    def __init__(self):
        self.svg_string = open('components/elements/arrow.svg', "rt").read()
        pygame_img = pygame.image.load(io.BytesIO(self.svg_string.encode()))
        self.size = pygame_img.get_size()

    def render(self, height):
        scaling_ratio = height / self.size[1]
        start = self.svg_string.find('<svg')
        scaled_svg_string = self.svg_string[:start + 4] + \
            f' transform="scale({scaling_ratio})"' + \
            self.svg_string[start + 4:]
        return pygame.image.load(io.BytesIO(scaled_svg_string.encode()))
