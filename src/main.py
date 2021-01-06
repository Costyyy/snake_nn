from random import random, choice, randrange, sample
from snake_env import Snake

from ple import PLE
from nNetwork import neural_net
import numpy as np


class Game:
    def __init__(self, batch_size=100, epochs=10, epsilon=1):
        self.p = Snake()
        self.model = neural_net()
        self.memory = []
        self.gamma = 0.95
        self.batch_size = batch_size
        self.epochs = epochs
        self.epsilon = epsilon

    def run(self):
        frame = 1
        # self.model.load_weights('my_model_weights.h5')
        done = True
        state = None
        while True:
            if done:
                state = np.array([self.p.reset()])

            # pick an action to do, either random or predicted
            if (random() < self.epsilon) and (frame < self.batch_size):
                action_index = choice([0, 1, 2, 3])
            else:
                # get action prediction from the model
                prediction = self.model.predict(state)
                action_index = np.argmax(prediction[0])
            # make move and memorize it
            new_state, reward, done, _ = self.p.step(action_index)
            new_state = np.array([new_state])
            self.memory.append((state, action_index, new_state, reward, done))
            state = new_state
            # if there is enough training data start training the NN
            if len(self.memory) >= self.batch_size:

                # pick a batch from the memory variable
                minibatch = sample(self.memory, self.batch_size)
                states = np.array([i[0] for i in minibatch])
                actions = np.array([i[1] for i in minibatch])
                next_states = np.array([i[2] for i in minibatch])
                rewards = np.array([i[3] for i in minibatch])
                dones = np.array([i[4] for i in minibatch])
                states = np.squeeze(states)
                next_states = np.squeeze(next_states)
                # adjust predicted actions using the obtained rewards, discarding predicted value if the game was lost
                targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1-dones)
                targets_full = self.model.predict_on_batch(states)
                ind = np.array([i for i in range(self.batch_size)])
                targets_full[[ind], [actions]] = targets
                # train model
                self.model.fit(states, targets_full, epochs=1, verbose=0)
                # decrease the chance of random moves as the time progresses
                if self.epsilon > 0.1:
                    self.epsilon -= (0.9 / self.epochs)


def main():
    game = Game()
    game.run()


main()
