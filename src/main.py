from random import random, choice, randrange

import pygame as pg
from ple import PLE
from ple.games.snake import Snake
from nNetwork import neural_net
import numpy as np


class Game:
    def __init__(self, batch_size=10, epoch=10, epsilon=1):
        self.height = self.width = 800
        game = Snake(width=self.height, height=self.width)
        self.p = PLE(game, fps=30, display_screen=True)
        self.model = neural_net()
        self.p.init()
        self.actions = self.p.getActionSet()
        self.memory = []
        self.gamma = 0.95
        self.batch_size = batch_size
        self.epochs = epoch
        self.epsilon = epsilon
        self.prev_dist = 0
        self.direction = -1

    def get_state(self):
        state_data = self.p.getGameState()
        # get coordinates disregarding screen size
        snake_x = state_data['snake_head_x'] / self.width
        snake_y = state_data['snake_head_y'] / self.height
        food_x = state_data['food_x'] / self.width
        food_y = state_data['food_y'] / self.height
        # check if there are walls in the immediate vicinity
        if snake_y >= .95:
            wall_up, wall_down = 1, 0
        elif snake_y < .05:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0
        if snake_x >= .95:
            wall_right, wall_left = 1, 0
        elif snake_x < .05:
            wall_right, wall_left = 0, 1
        else:
            wall_right, wall_left = 0, 0
        # check if there are body parts in the immediate vicinity
        body_down = 0
        body_up = 0
        body_left = 0
        body_right = 0
        if len(state_data['snake_body_pos']) > 3:
            for body_pos, body_dist in zip(state_data['snake_body_pos'][3:], state_data['snake_body'][3:]):
                if body_dist / self.width <= 0.020:
                    if body_pos[1] / self.width < snake_y:
                        body_down = 1
                    elif body_pos[1] / self.width > snake_y:
                        body_up = 1
                    if body_pos[0] / self.width < snake_x:
                        body_left = 1
                    elif body_pos[0] / self.width > snake_x:
                        body_right = 1
        # compile the state
        # food is up?(0, 1), food is to the right?(0, 1), food is down?(0, 1), food is to the left?(0,1)
        # is there any obstacle up, right, down, left?
        # is the direction up, left, right, down?
        state = [int(snake_y < food_y), int(snake_x < food_x),
                     int(snake_y > food_y), int(snake_x > food_x),
                 int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down),
                 int(wall_left or body_left), int(self.direction == 0), int(self.direction == 1),
                 int(self.direction == 2), int(self.direction == 3)]
        return state, (snake_x, snake_y, food_x, food_y)

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1 / 2

    def get_reward(self, reward):
        result = 0
        # if the game ended give a -500 reward
        if self.p.game_over():
            result = -500
        # if the snake ate the food give +100 reward
        elif reward == 1:
            result = 100
        elif reward == 0:
            _, coords = self.get_state()
            dist = self.get_distance(coords[:2], coords[2:])
            # if the snake is getting farther from the food give -1 rewards otherwise give 1
            if dist > self.prev_dist:
                result = -1
            elif dist < self.prev_dist:
                result = 1
            else:
                result = 0
            self.prev_dist = dist
        return result

    def set_dir(self, action):
        # memorize the direction of movement
        if action == 0 and self.direction != 3:
            self.direction = 0
        elif action == 1 and self.direction != 2:
            self.direction = 1
        elif action == 2 and self.direction != 1:
            self.direction = 2
        elif action == 3 and self.direction != 0:
            self.direction = 3

    def run(self):
        frame = 1
        while True:
            if self.p.game_over():
                self.p.reset_game()
                frame = 1
                self.memory = []
            # decrease the chance of random moves as the time progresses
            if self.epsilon > 0.1:
                self.epsilon -= (0.9 / self.epochs)

            # pick an action to do, either random or predicted
            if (random() < self.epsilon) and (frame < self.batch_size):
                action_index = choice([0, 1, 2, 3])
                action = self.actions[action_index]  # take a random direction
            else:
                # get action prediction from the model
                state, _ = self.get_state()
                prediction = self.model.predict(np.array([state])).flatten().tolist()

                action_index = prediction.index(max(prediction))
                action = self.actions[action_index]
            # make move and memorize it
            state, _ = self.get_state()
            reward = self.p.act(action)
            self.set_dir(action_index)
            reward = self.get_reward(reward)
            new_state = self.get_state()
            self.memory.append((state, action_index, new_state, reward, self.p.game_over()))
            # if we reached the required batch size train model
            if frame == self.batch_size:
                # get data from memory variable
                states = np.array([i[0] for i in self.memory])
                actions = np.array([i[1] for i in self.memory])
                next_states = np.array([i[2] for i in self.memory])
                rewards = np.array([i[3] for i in self.memory])
                dones = np.array([i[4] for i in self.memory])
                states = np.squeeze(states)
                next_states = np.squeeze(next_states)
                # adjust predictions using the obtained rewards
                targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1-dones)
                targets_full = self.model.predict_on_batch(states)
                ind = np.array([i for i in range(self.batch_size)])
                targets_full[[ind], [actions]] = targets
                # train model
                self.model.fit(states, targets_full, epochs=1, verbose=0)

                frame = 0
                self.memory = []
            frame += 1


def main():
    game = Game()
    game.run()


main()
