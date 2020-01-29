import numpy as np
import tensorflow as tf
from board_class import Board

#restore trained network
sess = tf.InteractiveSession()
meta_graph = tf.train.import_meta_graph('tic_tac_toe_model_nobad.meta')
meta_graph.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
states = graph.get_tensor_by_name('states:0')

#we don't want the network to be different for X and O, so we make each player see the board as X would
def state_from_board(board, counter):
    state = np.array(board.board)
    if counter == 1:
        state = -state
    state = state.reshape((1,9))
    return state

#returns an index based on current board configuration
def ai_pick(board, counter):
    state = state_from_board(board, counter)
    values = sess.run('action_values:0', feed_dict={states: state})
    max_val = values[0, 0]
    best_choice = 0
    for choice in range(9):
        if state[0, choice] == 0 and (values[0, choice] > max_val or state[0, best_choice] != 0):
            best_choice = choice
            max_val = values[0, choice]
    return best_choice

def random_pick(board, counter):
    state = state_from_board(board, counter)
    random_policy = np.array([1/9 for i in range(9)])
    #here we make sure invalid moves aren't made
    for i in range(9):
        if state[0, i] != 0:
            random_policy[i] = 0
    probsum = np.sum(random_policy)
    random_policy /= probsum
    return np.random.choice(9, p=random_policy)


agent_num = 0
agent_winsX = 0
agent_winsO = 0
drawsX = 0
drawsO = 0
symbols = ['X', 'O']
for game in range(8000):
    if game%100 == 0:
        print(game)
    winner = ''
    counter = 0
    board = Board()
    while winner == '':
        if counter == agent_num:
            index = ai_pick(board, counter)
        else:
            index = random_pick(board, counter)
        winner = board.setSquare(index, symbols[counter])
        counter = (counter + 1)%2
    counter = (counter + 1)%2
    if winner == 'D':
        if agent_num == 0:
            drawsX += 1
        else:
            drawsO += 1
    elif counter == agent_num:
        if counter == 0:
            agent_winsX += 1
        else:
            agent_winsO += 1
    agent_num = (agent_num + 1)%2

print('The trained agent won ' + str(agent_winsX) + ' times out of 4000 as X. It had ' + str(drawsX) + ' draws')
print('The trained agent won ' + str(agent_winsO) + ' times out of 4000 as O. It had ' + str(drawsO) + ' draws')
