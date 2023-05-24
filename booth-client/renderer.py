import pygame

from utils import *
from ui import *
from colors import *

import yaml

config_stream = open("config.yml", 'r')
config = yaml.load(config_stream, Loader=yaml.FullLoader)

DISPLAY_WIDTH = config['display']['width']
DISPLAY_HEIGHT = config['display']['height']

ASPECT_RATIO = config['display']['aspect_ratio'].split('_')
ASPECT_RATIO = [int(x) for x in ASPECT_RATIO]

OPTOTYPES_NUM = config['testing']['optotypes_num']

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

    def render(self, pointed_idx, current_results, which_test):
        self.line.render(self.display)

        center_x = DISPLAY_WIDTH / 2
        center_y = DISPLAY_HEIGHT / 2

        pointed_idx = min(pointed_idx, OPTOTYPES_NUM - 1)

        # Arrow
        if pointed_idx >= 0:
            arrow = Arrow()
            arrow_img = arrow.render(get_figure_size(
                self.logMAR * 1.2, self.distance, OPTOTYPES_NUM))

            self.display.blit(arrow_img,
                              (center_x - self.line.line_width / 2 + self.line.figure_offsets[pointed_idx] - arrow_img.get_size()[0] / 2,
                               center_y - get_figure_size(self.logMAR * 1.2, self.distance, 8.6)))

        # Corrects
        for idx, is_correct in enumerate(current_results):
            check_cross = CheckCross()
            check_cross_size = DISPLAY_HEIGHT / 6
            check_cross_gap = DISPLAY_HEIGHT / 15
            check_cross_img = check_cross.render(is_correct, DISPLAY_HEIGHT / 6)
            self.display.blit(check_cross_img,
                          (center_x - (check_cross_size*OPTOTYPES_NUM + check_cross_gap*(OPTOTYPES_NUM-1))/2 + (check_cross_size + check_cross_gap) * idx,
                           center_y + DISPLAY_HEIGHT / 5))

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

        text = small_font.render(f'Testing type : {config["testing"]["type"]} ({which_test})', True, (0, 0, 0))
        self.display.blit(text, (10, 130))

        if pointed_idx >= 0:
            str_pointed = 'Pointed Figure: ' + \
                self.line.figures[pointed_idx].name + \
                '   |  ( Index = ' + str(pointed_idx + 1) + ' )'
            text = small_font.render(str_pointed, True, (0, 0, 0))
            self.display.blit(text, (10, 160))

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

    def render_result(self, logMAR):
        center_x = DISPLAY_WIDTH / 2
        center_y = DISPLAY_HEIGHT / 2

        str_logMAR = 'Result logMAR: ' + str(logMAR)
        str_snellen = 'Result Snellen: ' + snellen_dict[logMAR]

        FONT = 'components/fonts/' + config['font']
        big_font = pygame.font.Font(FONT, 50)

        text_logMAR = big_font.render(str_logMAR, True, (0, 0, 0))
        text_width = text_logMAR.get_size()[0]

        text_snellen = big_font.render(str_snellen, True, (0, 0, 0))
        text_width = max(text_width, text_snellen.get_size()[0])

        self.display.blit(text_logMAR, (center_x - text_width/2, center_y))
        self.display.blit(text_snellen, (center_x - text_width/2, center_y + 70))

class DisplayScreen:
    def __init__(self, distance, display):
        self.distance = distance
        self.display = display
        
    def render(self, header_str, msg_str):
        self.display.fill((255, 255, 255))
        center_x = DISPLAY_WIDTH / 2
        center_y = DISPLAY_HEIGHT / 2

        FONT = 'components/fonts/' + config['font']
        big_font = pygame.font.Font(FONT, 70)
        small_font = pygame.font.Font(FONT, 35)

        text_header = big_font.render(header_str, True, (0, 0, 0))
        text_width = text_header.get_size()[0]

        self.display.blit(text_header, (center_x - text_width/2, center_y - 70))

        text_msg = small_font.render(msg_str, True, (0, 0, 0))
        text_width = text_msg.get_size()[0]

        self.display.blit(text_msg, (center_x - text_width/2, center_y + 20))


class OptionScreen:
    def __init__(self, display):
        self.display = display
        self.options = {
            'Display Size (Inches)' : ['13.3', '14.0', '15.6', '17.3', '19.0', '21.5', '23.0', '24.0', '27.0', '32.0', '34.0', '38.0', '43.0', '49.0', '55.0', '65.0', '70.0', '75.0', '80.0', '85.0', '86.0', '90.0'],
            'Resolution' : [ '1280 x 720', '1600 x 900', '1920 x 1080', '2560 x 1440', '3840 x 2160', '7680 x 4320'],
            'Testing Type' : [ 'Single Eye Both Sides', 'Single Eye Left Side', 'Single Eye Right Side', 'Both Eyes Together'], 
            'Testing Distance (Meters)' : ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0'],
            # 'Optotypes per Set' : ['1', '2', '3', '4', '5'],
            'Run Introduction Video' : ['Yes', 'No']
        }

        self.current_options = {
            'Display Size (Inches)' : str(config['display']['screen_size']),
            'Resolution' : f'{str(config["display"]["width"])} x {str(config["display"]["height"])}',
            'Testing Type' : str(config['testing']['type']), 
            'Testing Distance (Meters)' : str(config['testing']['distance']),
            # 'Optotypes per Set' : str(config['testing']['optotypes_num']),
            'Run Introduction Video' : 'Yes' if config['run_intro'] else 'No'
        }

        for option_name in self.current_options:
            selected_option = self.current_options[option_name]
            if not (str(selected_option) in self.options[option_name]):
                self.current_options[option_name] = f'{selected_option} - Custom'
                self.options[option_name] = [self.current_options[option_name]] + self.options[option_name]

        FONT = 'components/fonts/' + config['font']
        small_font = pygame.font.Font(FONT, 35)

        self.max_option_name_width = -1
        self.max_option_selection_width = -1
        for option_name in self.options:
            selections = self.options[option_name]
            text_msg = small_font.render(option_name, True, Color.black)
            text_width = text_msg.get_size()[0]
            self.max_option_name_width = max(self.max_option_name_width, text_width)

            for selection_name in selections:
                text_msg = small_font.render(str(selection_name), True, Color.black)
                text_width = text_msg.get_size()[0]
                self.max_option_selection_width = max(self.max_option_selection_width, text_width)

        self.max_option_width = self.max_option_name_width + self.max_option_selection_width

        self.max_option_height = 45 * len(self.options)

    def render(self, selected_index, selection_input):

        self.display.fill((255, 255, 255))
        center_x = DISPLAY_WIDTH / 2
        center_y = DISPLAY_HEIGHT / 2

        FONT = 'components/fonts/' + config['font']
        big_font = pygame.font.Font(FONT, 70)
        small_font = pygame.font.Font(FONT, 35)

        arrow_img = pygame.image.load('components/elements/right-arrow.png')

        text_header = big_font.render('Options', True, Color.black)
        text_width = text_header.get_size()[0]

        self.display.blit(text_header, (center_x - text_width/2, center_y - 70 - self.max_option_height/2))

        
        current_line = 0
        for option_name in self.options:
            selections = self.options[option_name]
            selected = str(self.current_options[option_name])

            if current_line == selected_index:
                current_option_index = selections.index(str(selected))
                if current_option_index < len(selections) - 1 and selection_input == 'right':
                    current_option_index += 1
                    selected = selections[current_option_index]
                    self.current_options[option_name] = selected
                elif current_option_index > 0 and selection_input == 'left':
                    current_option_index -= 1
                    selected = selections[current_option_index]
                    self.current_options[option_name] = selected

                selected = str(selected)
                self.display.blit(arrow_img, (center_x - self.max_option_width/2 - 45, center_y + 35 - self.max_option_height/2 + current_line * 45))

            text_msg = small_font.render(option_name, True, (0, 0, 0))
            self.display.blit(text_msg, (center_x - self.max_option_width/2, center_y + 30 - self.max_option_height/2 + current_line * 45))

            text_msg = small_font.render(selected, True, (0, 0, 0))
            self.display.blit(text_msg, (center_x - self.max_option_width/2 + self.max_option_name_width + 40, center_y + 30 - self.max_option_height/2 + current_line * 45))


            current_line += 1

        if selected_index == len(self.options):
            self.display.blit(arrow_img, (center_x - self.max_option_width/2 - 45, center_y + 35 - self.max_option_height/2 + current_line * 45 + 40))
        
        text_msg = small_font.render('Exit', True, (0, 0, 0))
        self.display.blit(text_msg, (center_x - self.max_option_width/2, center_y + 30 - self.max_option_height/2 + current_line * 45 + 40))

        current_line += 1

        if selected_index == len(self.options) + 1:
            self.display.blit(arrow_img, (center_x - self.max_option_width/2 - 45, center_y + 35 - self.max_option_height/2 + current_line * 45 + 40))

        text_msg = small_font.render('Save and Exit (Requires restart)', True, (0, 0, 0))
        self.display.blit(text_msg, (center_x - self.max_option_width/2, center_y + 30 - self.max_option_height/2 + current_line * 45 + 40))

        
    def save(self):
        config['display']['width'] = int(self.current_options['Resolution'].replace(' - Custom ', '').split(' x ')[0])
        config['display']['height'] = int(self.current_options['Resolution'].replace(' - Custom', '').split(' x ')[1])

        config['display']['screen_size'] = round(float(self.current_options['Display Size (Inches)'].replace(' - Custom', '')), 1)

        config['testing']['type'] = self.current_options['Testing Type'].replace(' - Custom', '')
        config['testing']['distance'] = round(float(self.current_options['Testing Distance (Meters)'].replace(' - Custom', '')), 1)
        # config['testing']['optotypes_num'] = int(self.current_options['Optotypes per Set'].replace(' - Custom', ''))

        config['run_intro'] = True if self.current_options['Run Introduction Video'].replace(' - Custom', '') == 'Yes' else False

        config_stream = open("config.yml", 'w')
        yaml.dump(config, config_stream)