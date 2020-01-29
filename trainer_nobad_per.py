import tensorflow as tf
from board_class import Board
from prioritized_memory_class import Memory
from dueling_model_class import Model
import numpy as np

gamma = 0.5#discount factor
batch_size = 200

#epsilon-greedy policy
def choose_action(epsilon, state, model, sess):
    r = np.random.random()
    #act randomly
    if r < epsilon or np.linalg.norm(state) == 0:
        choice = np.random.randint(0,9)
        while state[0, choice] != 0:
            choice = (choice + 1)%9
        return choice
    else:#follow policy
        values = model.predict_one(state, sess)
        max_val = values[0, 0]
        best_choice = 0
        for choice in range(9):
            if state[0, choice] == 0 and (values[0, choice] > max_val or state[0, best_choice] != 0):
                best_choice = choice
                max_val = values[0, choice]
        return best_choice

def get_max_reward(state, model, sess):
    values = model.predict_one(state, sess)
    max_val = values[0, 0]
    best_choice = 0
    for choice in range(9):
        if state[0, choice] == 0 and (values[0, choice] > max_val or state[0, best_choice] != 0):
            best_choice = choice
            max_val = values[0, choice]
    return max_val

#we don't want the network to be different for X and O, so we make each player see the board as X would
def state_from_board(board, counter):
    state = np.array(board.board)
    if counter == 1:
        state = -state
    state = state.reshape((1,9))
    return state

memory = Memory(3000)
model = Model(0.0001)

epsilon = 0.9
with tf.Session() as sess:
    sess.run(model.var_init)
    for game in range(30000):
        if game%100 == 0:
            print(game)
        board = Board()
        winner = ''
        counter = 0
        symbols = ['X', 'O']
        #we need to store samples temporarily because we don't get their values till the end of each game
        samples = []#each sample contains state, action, reward, and next state
        while winner == '':
            state = state_from_board(board, counter)

            action = choose_action(epsilon, state, model, sess)

            current_sample = []
            current_sample.append(state)
            current_sample.append(action)

            winner = board.setSquare(action, symbols[counter])
            current_sample.append(0.5)#placeholder reward. we change this when we know the winner

            samples.append(current_sample)
            #switch to next player
            counter = (counter + 1)%2

        #lol this is so ugly
        xreward = 0
        if winner == 'X':
            xreward = 0.5
        elif winner == 'O':
            xreward = -0.5

        #add the next state to each sample and set rewards based on winner
        num_samples = len(samples)
        for i in range(num_samples):
            #next state
            if i < num_samples - 2:
                samples[i].append(samples[i + 2][0])
            else:
                samples[i].append(None)

            if i%2 == 0:
                samples[i][2] = samples[i][2] + xreward*(i+1)/num_samples
            else:
                samples[i][2] = samples[i][2] - xreward*(i+1)/num_samples

            error = (model.predict_one(samples[i][0], sess)[0, samples[i][1]] - samples[i][2]) ** 2
            memory.add_sample(samples[i], error)

        alpha = epsilon #because why not?
        sample_batch = memory.sample_samples(batch_size, alpha)

        actual_batch_size = len(sample_batch)
        state_batch = np.zeros((actual_batch_size, 9))
        next_state_batch = np.zeros((actual_batch_size, 9))
        action_batch = [sample[1] for sample in sample_batch]

        for i, sample in enumerate(sample_batch):
            state_batch[i] = sample[0]
            if sample[3] is not None:
                next_state_batch[i] = sample[3]

        qsa_batch = model.predict_batch(state_batch, sess)

        for i in range(actual_batch_size):
            qsa_batch[i, action_batch[i]] = sample_batch[i][2]
            if sample_batch[i][3] is not None:
                qsa_batch[i, action_batch[i]] += gamma*get_max_reward(state_batch[i].reshape((1, 9)), model, sess)

        model.train_batch(state_batch, qsa_batch, sess)

        epsilon = 0.9 * np.exp(-0.0001 * game)
    model.save(sess, 'tic_tac_toe_model_nobad_improved')
    model.plot_losses('losses_improved.png')