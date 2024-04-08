def minimax(total, turn, aplha, beta):

    if total == 20:
        return 0
    elif total > 20:
        if turn:
            return -1
        else:
            return 1

    if turn:
        max_eval = -float('inf')

        for i in range(1,4):
            eval = minimax(total+i, False, aplha, beta)
            max_eval = max(max_eval, eval)
            aplha = max(max_eval, eval)

            if beta <= aplha:
                break

        return max_eval
    else:
        min_eval = float('inf')
        for i in range(1,4): 
            eval = minimax(total+i, True, aplha, beta) 
            min_eval = min(min_eval, eval)
            beta = min( beta, eval)

            if beta <= aplha:
                break

        return min_eval



total = 0

while True: 
    human_move = int(input('Enter integer in [1,2,3]: '))
    while human_move not in [1,2,3]:
        print('Invalid move')
        human_move = int(input('Enter integer in [1,2,3]: ')) 

    total += human_move

    if total >= 20:
        print('Yow win!')
        break

    print(f"Total = {total}")
    print("AI is making move....")
    max_eval = -float('inf')
    ai_move = 1
    for i in [1, 2, 3]:
        eval = minimax(total+i, False, -float('inf'), float('inf'))

        if eval > max_eval:
            max_eval = eval
            ai_move = i
    
    total += ai_move

    print(f'AI added {ai_move}, total is: {total}')

    if total >= 20:
        print('AI wins!')
        break


