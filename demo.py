import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import yaml
import math

from optotypes import *
from renderer import *
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

corrects = {}
for logMAR in snellen_dict.keys():
    corrects[logMAR] = []
print(corrects)

if __name__ == '__main__':
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Visual Acuity Test - Display')

    optotypes = load_optotypes()

    crashed = False
    cur_logMAR = 1
    cur_optotype = 3
    cur_pointed = -1
    update_screen = True
    update_optotypes = True


    test = TestScreen(list(optotypes.values())[cur_optotype], 2, 5, cur_logMAR, display)
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if cur_logMAR < 1:
                        cur_logMAR = round((cur_logMAR + 0.1) * 10) / 10
                        update_screen = True
                        update_optotypes = True
                if event.key == pygame.K_DOWN:
                    if cur_logMAR > -0.3:
                        cur_logMAR = round((cur_logMAR - 0.1) * 10) / 10 
                        update_screen = True
                        update_optotypes = True

                if event.key == pygame.K_LEFT:
                    if cur_optotype > 0:
                        cur_optotype -= 1
                        update_screen = True
                        update_optotypes = True
                if event.key == pygame.K_RIGHT:
                    if cur_optotype < len(optotypes) - 1:
                        cur_optotype += 1
                        update_screen = True
                        update_optotypes = True

                if event.key == pygame.K_SPACE:
                    update_screen = True
                    update_optotypes = True

                if event.key == pygame.K_a:
                    if cur_pointed > -1:
                        cur_pointed -= 1
                    update_screen = True
                if event.key == pygame.K_d:
                    if cur_pointed < len(optotypes) - 1:
                        cur_pointed += 1
                    else:
                        cur_pointed = 0
                    update_screen = True

        if update_screen:
            display.fill((255, 255, 255))
            if update_optotypes:
                test = TestScreen(list(optotypes.values())[cur_optotype], 2, 5, cur_logMAR, display)
                update_optotypes = False
            test.render(cur_pointed)
            update_screen = False
            print([x.name for x in test.line.figures])
        pygame.display.update()
