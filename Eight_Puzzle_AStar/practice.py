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

    def is_goal(self, current_state):
        return np.array_equal(current_state, self.goal_state)

    def print_state(self, state):
        print(state[: ,:])


    def get_possible_moves(self, current_state):
        possible_moves = []
        blank_pos = np.where(current_state == 0)
        directions = [(0, -1),(0 , 1), (-1 , 0), (1 , 0)]
        # check for boundaries
        for dir in directions:
            new_pos = (blank_pos[0] + dir[0], blank_pos[1] + dir[1])
            if 0 <= new_pos[0] < 3 and 0 <=  new_pos[1] < 3:  
                new_state = np.copy(current_state)
                new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
                possible_moves.append(new_state)
        
        return possible_moves

    def heuristic(self, cur_state):
        return np.count_nonzero(cur_state != self.goal_state)

    def Solve(self):
        queue = PriorityQueue()
        initial_state = State(self.initial_state, None)
        queue.put((0, initial_state))
        visited = set()

        while not queue.empty():
            priority, cur_state = queue.get()

            if self.is_goal(cur_state.state):
                return cur_state
            
            for move in self.get_possible_moves(cur_state.state):
                move_state = State(move, cur_state)
       
                
                if str(move_state.state) not in visited:
                    priority = self.heuristic(move_state)
                    visited.add(str(move_state.state))
                    queue.put((priority, move_state))
        return None

initial_state = []
goal_state = []
print('Enter the initial State: ')

for _ in range(3):
    row = input().split()
    initial_state.append(row)

initial_state = np.array(initial_state)

print('Enter the goal State: ')


for _ in range(3):
    row = input().split()
    goal_state.append(row)

goal_state = np.array(goal_state)

puzzle = Puzzle(initial_state, goal_state)

solution = puzzle.Solve()

if solution is not None:
    moves = []
    while solution is not None:
        moves.append(solution.state)
        solution = solution.parent
    for move in reversed(moves):
        puzzle.print_state(move)
else:
    print('No solution')
