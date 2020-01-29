import tensorflow as tf
from board_class import Board
from memory_class import Memory
from model_class import Model
import numpy as np

gamma = 0.5#discount factor
batch_size = 200

#epsilon-greedy policy
def choose_action(epsilon, state, model, sess):
    r = np.random.random()
    #act randomly
    if r < epsilon or np.linalg.norm(state) == 0:
        return np.random.randint(0,9)
    else:#follow policy
        return np.argmax(model.predict_one(state, sess))

#we don't want the network to be different for X and O, so we make each player see the board as X would
def state_from_board(board, counter):
    state = np.array(board.board)
    if counter == 1:
        state = -state
    state = state.reshape((1,9))
    return state    
    
memory = Memory(3000)#not sure how many samples will crash the computer, so let's keep it conservative for now
model = Model()

epsilon = 0.9
#we'll play a thousand games
with tf.Session() as sess:
    sess.run(model.var_init)
    for game in range(4000):
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
            
            if board.getSquare(action) == 0:
                winner = board.setSquare(action, symbols[counter])
                current_sample.append(1)#placeholder reward. we change this when we know the winner
            else:
                winner = 'ERR'
                current_sample.append(-2)#if an invalid move was made, give a bad reward

            samples.append(current_sample)
            #switch to next player
            counter = (counter + 1)%2

        #this loop adds the next state to each sample
        num_samples = len(samples)
        for i in range(num_samples):
            if i < num_samples - 2:
                samples[i].append(samples[i + 2][0])
            else:
                samples[i].append(None)
            memory.add_sample(samples[i])

        sample_batch = memory.sample_samples(batch_size)
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
            for choice in range(9):
                if state_batch[i, choice] != 0:
                    qsa_batch[i, choice] = -2
            if sample_batch[i][3] is None:
                qsa_batch[i, action_batch[i]] = sample_batch[i][2]
            else:
                qsa_batch[i, action_batch[i]] = sample_batch[i][2] + gamma*np.amax(model.predict_one(next_state_batch[i].reshape((1,9)), sess))
            
        model.train_batch(state_batch, qsa_batch, sess)
        
        epsilon = 0.9*np.exp(-0.001*game)
    model.save(sess)
    model.plot_losses()
