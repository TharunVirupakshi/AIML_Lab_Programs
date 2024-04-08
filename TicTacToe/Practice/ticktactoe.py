class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.player = 'X'

    def printBoard(self):
        for row in self.board:
            print(' | '.join(row))
            print('-'*10)

    def is_draw(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def is_game_over(self):

        for row in self.board:
            if row.count(row[0]) == len(row) and row[0] != ' ':
                return row[0]
        
        for col in range(len(self.board[0])):
            check = []
            for row in self.board:
                check.append(row[col])
            if check.count(check[0]) == len(check) and check[0] != ' ':
                return check[0]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        return False

    def dfs(self, board, depth, player):

        winner = self.is_game_over()

        if winner:
            if winner == 'X':
                return {'score': 1}
            else:
                return {'score': -1}
        elif self.is_draw():
                return {'score' : 0}

        if player == 'X':
            best = {'score' : -float('inf')}
            symbol = 'X'
        else:
            best = {'score' : float('inf')}
            symbol = '0'
        
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = symbol
                    score = self.dfs(board, depth + 1, '0' if player=='X' else 'X')
                    board[i][j] = ' '
                    score['row'] = i
                    score['col'] = j

                    if player == 'X':
                        if score['score'] > best['score']:
                            best = score
                    else:
                       if score['score'] < best['score']:
                            best = score 
        return best

    def play(self):
        while True:
            self.printBoard()

            winner  = self.is_game_over()

            if winner or self.is_draw():
                print('GAME OVER!')
                if self.is_draw():
                    print("Its a draw.")
                else:
                    print(f"Player {winner} wins!")
                break
            
            if self.player == "X":
                best_move = self.dfs(self.board, 0, 'X')
                self.board[best_move['row']][best_move['col']] = 'X'
            else:
                while True:
                    try: 
                        row = int(input('Enter row: '))
                        col = int(input('Enter col: '))

                        if self.board[row][col] == ' ':
                            self.board[row][col] = '0'
                            break
                        else:
                            print('Invalid move!')
                    except (ValueError, IndexError):
                        print('Invalid input')
                        

            self.player = '0' if self.player=='X' else 'X' 

                

game = TicTacToe()
game.play()