import numpy as np
from queue import PriorityQueue

class State:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __lt__(self, other):
        return False

class Puzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def printState(self, state):
        print(state[:,:])

    def is_goal(self, state):
        return np.array_equal(state, self.goal_state)

    def get_possible_moves(self, state):
        possible_moves = []
        zero_pos = np.where(state == 0)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] 

        for dir in directions:
            new_pos = (zero_pos[0] + dir[0], zero_pos[1] + dir[1])
            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
                new_state = np.copy(state)
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos] , new_state[zero_pos]

                possible_moves.append(new_state)

        return possible_moves

    def heuristic(self, state):
        return np.count_nonzero(state != self.goal_state)

    def solve(self):
        queue = PriorityQueue()
        initial_state = State(self.initial_state, None)
        queue.put((0, initial_state))

        visited = set()

        while not queue.empty():
            priority, cur_state = queue.get()

            if self.is_goal(cur_state.state ):
                return cur_state
            
            for move in self.get_possible_moves(cur_state.state):
                move_state = State(move, cur_state)

                if str(move_state.state) not in visited:
                    visited.add(str(move_state.state))

                    priority = self.heuristic(move_state.state)

                    queue.put((priority, move_state))
        return None


# No solution
# init_state = np.array([[2,3,4],
#               [5,7,6],
#               [1,8,0]])
# goal_state =np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 0]])

init_state = np.array([[0,1,2],
                      [3,4,5],
                      [7,8,6]])
goal_state = np.array([[1,2,3],
                      [4,5,0],
                      [7,8,6]])

puzzle = Puzzle(initial_state = init_state, goal_state = goal_state)

solution = puzzle.solve()

if solution is not None:
    moves = []

    while solution is not None:
        moves.append(solution.state)
        solution = solution.parent
    
    for move in reversed(moves):
        puzzle.printState(move)
else:
    print('No Solution!')

