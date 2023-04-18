import os
import io
import random
import pygame
import yaml


class Figure:
    def __init__(self, name, svg_string):
        self.name = name
        self.svg_string = svg_string
        pygame_img = pygame.image.load(io.BytesIO(svg_string.encode()))
        self.size = pygame_img.get_size()

    # Returns a pygame surface with the given height in pixels
    def render(self, height):
        scaling_ratio = height / self.size[1]
        start = self.svg_string.find('<svg')
        scaled_svg_string = self.svg_string[:start + 4] + \
            f' transform="scale({scaling_ratio})"' + \
            self.svg_string[start + 4:]
        return pygame.image.load(io.BytesIO(scaled_svg_string.encode()))


class Optotype:
    def __init__(self, name, figures, height):
        self.name = name
        self.figures = figures
        self.height = height

    # Returns a list of randomly chosen figures.
    #
    # The chosen figures are unique if the inputted number is less than
    # the number of figures in the optotype set, otherwise duplicates would be present
    def choose_random(self, count):
        random_list = []
        # Test if there is enough figure in the optotype set
        try:
            random_list = random.sample(list(self.figures.values()), count)
        except ValueError:
            random_list.append(random.choice(list(self.figures.values())))
            for i in range(count - 1):
                rand_figure = random.choice(list(self.figures.values()))
                # Check and regenerate to keep any 3 consecutive figures unique
                while rand_figure == random_list[len(random_list) - 1] or rand_figure == random_list[max(0, len(random_list) - 2)]:
                    rand_figure = random.choice(list(self.figures.values()))
                random_list.append(rand_figure)
        return random_list


# Load Optotypes
def load_optotypes():
    base_dir = os.getcwd()
    base_dir = os.path.join(base_dir, 'components/optotypes')
    optotypes = {}
    for optotype_name in os.listdir(base_dir):
        optotype_path = os.path.join(base_dir, optotype_name)
        figures = {}

        # Load config file for each optotypes
        config_path = os.path.join(optotype_path, 'config.yml')
        config_stream = open(config_path, 'r')
        config = yaml.load(config_stream, Loader=yaml.FullLoader)

        for figure in os.listdir(optotype_path):
            if os.path.splitext(figure)[0] != 'config':
                figure_path = os.path.join(optotype_path, figure)
                svg_string = open(figure_path, "rt").read()

                figure_name = os.path.splitext(figure)[0]
                figures[figure_name] = Figure(figure_name, svg_string)

        optotypes[optotype_name] = Optotype(
            optotype_name, figures, config['height'])
    return optotypes
