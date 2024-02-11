###############################################################################
# This file contains helper functions and the heuristic functions
# for our AI agents to play the Mancala game.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import sys

###############################################################################
### DO NOT MODIFY THE CODE BELOW

### Global Constants ###
TOP = 0
BOTTOM = 1

### Errors ###
class InvalidMoveError(RuntimeError):
    pass

class AiTimeoutError(RuntimeError):
    pass

### Functions ###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_opponent(player):
    if player == BOTTOM:
        return TOP
    return BOTTOM

### DO NOT MODIFY THE CODE ABOVE
###############################################################################


def heuristic_basic(board, player):
    """
    Compute the heuristic value of the current board for the current player 
    based on the basic heuristic function.

    :param board: the current board.
    :param player: the current player.
    :return: an estimated utility of the current board for the current player.
    """
    mancala_top = board.mancalas[TOP]
    mancala_bot = board.mancalas[BOTTOM]

    if player == TOP:
        return mancala_top - mancala_bot
    
    else:
        return mancala_bot - mancala_top
    
    #raise NotImplementedError


def heuristic_advanced(board, player): 
    """
    Compute the heuristic value of the current board for the current player
    based on the advanced heuristic function.

    :param board: the current board object.
    :param player: the current player.
    :return: an estimated heuristic value of the current board for the current player.
    """
    mancala_top = board.mancalas[TOP]
    mancala_bot = board.mancalas[BOTTOM]
    pockets_top = board.pockets[TOP]
    pockets_bot = board.pockets[BOTTOM]
    
    # Checking if the player can capture opponent's pocket
    


    if player == TOP:

        heuristic_value = (mancala_top - mancala_bot) * 1.5

        heuristic_value += (sum(pockets_top) - sum(pockets_bot))

        for i in range(board.dimension):
            if pockets_top[i] == 0 and pockets_bot[board.dimension - 1 - i] != 0:
                for j in range(board.dimension):
                    if pockets_top[j] == abs(i - j) and i != j:
                        heuristic_value += pockets_bot[board.dimension - 1 - i]/2

            if i != board.dimension - 1:
                if pockets_top[i] >= pockets_top[i+1]:
                    heuristic_value += 1
                
                else:
                    heuristic_value -= 0.5
            
            if pockets_top[i] == 0 or pockets_top[i] > 0.5 * sum(pockets_top):
                heuristic_value -= 0.25 * (sum(pockets_top) - pockets_top[i])
            
            if pockets_bot[i] == 0 and pockets_top[board.dimension - 1 - i] != 0:
                for j in range(board.dimension):
                    if pockets_bot[j] == abs(i - j) and i != j:
                        heuristic_value -= pockets_top[board.dimension - 1 - i]/4

        return heuristic_value
    
    else:

        heuristic_value = (mancala_bot - mancala_top) * 1.5

        heuristic_value += (sum(pockets_bot) - sum(pockets_top))


        for i in range(board.dimension):
            if pockets_bot[i] == 0 and pockets_top[board.dimension - 1 - i] != 0:
                for j in range(board.dimension):
                    if pockets_bot[j] == abs(i - j):
                        heuristic_value += pockets_top[board.dimension - 1 - i]/2
            
            if i != board.dimension - 1:
                if pockets_bot[i] >= pockets_bot[i+1]:
                    heuristic_value += 1
                
                else:
                    heuristic_value -= 0.5
            
            if pockets_bot[i] == 0 or pockets_bot[i] > 0.5 * sum(pockets_bot):
                heuristic_value -= 0.25 * (sum(pockets_bot) - pockets_bot[i])

            if pockets_top[i] == 0 and pockets_bot[board.dimension - 1 - i] != 0:
                for j in range(board.dimension):
                    if pockets_top[j] == abs(i - j):
                        heuristic_value -= pockets_bot[board.dimension - 1 - i]/4

        return heuristic_value
    
    # raise NotImplementedError
