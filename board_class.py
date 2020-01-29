import math

class Board:
    def __init__(self):
        #store board as a nine-element list
        #0 indicates that the space is empty
        self.board = [0 for i in range(9)]
        #there are 8 ways/regions to win, i.e. 3 rows, 3 cols, 2 diags
        #every time something is placed in that region, change the region's val
        #if the absolute value is 3, someone won
        self.vals = [0 for i in range(8)]

    def checkWinner(self):
        for val in self.vals:
            if val == -3:
                return 'X'
            elif val == 3:
                return 'O'
        if 0 not in self.board:
            return 'D'#draw
        return ''#no winner yet

    def setSquare(self, index, symbol):
        #represent X with -1 and O with 1
        
        if symbol == 'X':
            v = -1
        else:
            v = 1

        self.board[index] = v

        row = math.floor(index/3)
        col = index%3
        self.vals[row] += v
        self.vals[3 + col] += v

        if col == row:
            self.vals[6] += v
        if col == 2 - row:
            self.vals[7] += v
        return self.checkWinner()

    def getSquare(self, index):
        return self.board[index]

    def printBoard(self):
        for i in range(9):
            chars = ['X', '-', 'O']
            if i%3 == 2:
                print(chars[self.board[i] + 1], end='\n')
            else:
                print(chars[self.board[i] + 1], end='')
        print('')
