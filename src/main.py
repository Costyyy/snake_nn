from random import random, choice, randrange

import pygame as pg
from ple import PLE
from ple.games.snake import Snake
from nNetwork import neural_net
import numpy as np


class Game:
    def __init__(self):
        game = Snake(width=800, height=800)
        self.p = PLE(game, fps=30, display_screen=True)
        self.model = neural_net([15, 16])
        self.p.init()
        self.reward = 0.0
        self.actions = self.p.getActionSet()

    def run(self):
        frame = 0
        while True:
            if self.p.game_over():
                self.p.reset_game()

            observation = self.p.getScreenRGB()
            if (random() < 1) and (frame < 10):
                action = self.actions[choice([0, 1, 2, 3])]  # take a random direction
            else:
                # get action prediction from the model
                state = self.p.getGameState()
                state = np.array([x for x in list(state.values())[:4]])
                prediction = self.model.predict(np.array([state])).flatten().tolist()
                action = self.actions[prediction.index(max(prediction))]

            reward = self.p.act(0)
            frame += 1


def main():
    game = Game()
    game.run()


main()
