from random import random, choice, randrange

import pygame as pg
from ple import PLE
from ple.games.snake import Snake
from nNetwork import neural_net
import numpy as np


class Game:
    def __init__(self, batch_size=10, epoch=10, epsilon=1):
        game = Snake(width=800, height=800)
        self.p = PLE(game, fps=30, display_screen=True)
        self.model = neural_net([15, 16])
        self.p.init()
        self.reward = 0.0
        self.actions = self.p.getActionSet()
        self.memory = []
        self.batch_size = batch_size
        self.epochs = epoch
        self.epsilon = epsilon
        self.prev_dist = 0

    def get_state(self):
        state = self.p.getGameState()
        return np.array([x for x in list(state.values())[:4]])

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x2 - x1) + abs(y2 - y1)

    def get_reward(self, reward):
        if reward == -5:
            return -100
        elif reward == 1:
            return 10
        elif reward == 0:
            state = self.get_state()
            dist = self.get_distance(state[:2], state[2:])
            if dist > self.prev_dist:
                return -10
            elif dist < self.prev_dist:
                return 10
            else:
                return 0

    def run(self):
        frame = 0
        while True:
            if self.p.game_over():
                self.p.reset_game()

            if self.epsilon > 0.1:
                self.epsilon -= (0.9 / self.epochs)

            observation = self.p.getScreenRGB()
            if (random() < self.epsilon) and (frame < self.batch_size):
                action_index = choice([0, 1, 2, 3])
                action = self.actions[action_index]  # take a random direction
            else:
                # get action prediction from the model
                state = self.get_state()
                prediction = self.model.predict(np.array([state])).flatten().tolist()
                # print(prediction)
                action_index = prediction.index(max(prediction))
                action = self.actions[action_index]

            state = self.get_state()
            prediction = self.model.predict(np.array([state])).flatten().tolist()
            reward = self.p.act(action)
            reward = self.get_reward(reward)
            prediction[action_index] = reward
            self.memory.append((state, prediction))
            # print(reward)
            if frame == 10:
                # get training set from experience
                x_train = []
                y_train = []
                for el in self.memory:
                    x_train.append(el[0])
                    y_train.append(el[1])

                loss = self.model.fit(np.array(x_train), np.array(y_train),
                                      batch_size=self.batch_size, epochs=self.epochs, verbose=0)
                # reset frames and expereince
                frame = 0
                self.memory = []
            frame += 1


def main():
    game = Game()
    game.run()


main()
