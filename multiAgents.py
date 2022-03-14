# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)

        return legalMoves[scores.index(bestScore)]

    def evaluationFunction(self, currentGameState, action):
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        food = currentGameState.getFood()
        ghostStates = currentGameState.getGhostStates()

        closestGhost = food.width * food.height    # max maze distance is every square
        for ghostState in ghostStates:
            dist = manhattanDistance(newPos, ghostState.getPosition())
            if dist < closestGhost: closestGhost = dist

        closestFood = food.width * food.height    # max maze distance is every square
        for x in range(food.width):
            for y in range(food.height):
                if currentGameState.hasFood(x, y):
                    dist = manhattanDistance(newPos, (x,y))
                    if dist < closestFood: closestFood = dist

        if closestGhost < 4:
            if action == Directions.STOP:
                return -sys.maxsize
            return successorGameState.getScore() - closestFood - closestGhost
        else:
            return successorGameState.getScore() - closestFood

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        
        legalMoves = gameState.getLegalActions()
        scores = [self.value(gameState.generateSuccessor(0, move), 1) for move in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        return legalMoves[scores.index(bestScore)]

    def value(self, gameState, depth):

        if depth == self.depth * gameState.getNumAgents(): return self.evaluationFunction(gameState)

        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        if depth % gameState.getNumAgents() == 0:  return self.max(gameState, depth + 1)
        else:   return self.min(gameState, depth + 1, depth % gameState.getNumAgents())

    def max(self, gameState, depth, agentIndex = 0):

        value = -sys.maxsize

        for action in gameState.getLegalActions(agentIndex):
            value = max(value, self.value(gameState.generateSuccessor(agentIndex, action), depth))

        return value

    def min(self, gameState, depth, agentIndex):
        
        value = sys.maxsize

        for action in gameState.getLegalActions(agentIndex):
            value = min(value, self.value(gameState.generateSuccessor(agentIndex, action), depth))

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        legalMoves = gameState.getLegalActions()

        alpha = -sys.maxsize
        scores = []
        for move in legalMoves:
            score = self.value(gameState.generateSuccessor(0, move), 1, alpha, sys.maxsize)
            if score > alpha: alpha = score
            scores.append(score)
            
        bestScore = max(scores)

        return legalMoves[scores.index(bestScore)]

    def value(self, gameState, depth, alpha, beta):

        if depth == self.depth * gameState.getNumAgents(): return self.evaluationFunction(gameState)

        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        if depth % gameState.getNumAgents() == 0: return self.max(gameState, depth + 1, alpha, beta)
        else: return self.min(gameState, depth + 1, alpha, beta, depth % gameState.getNumAgents())

    def max(self, gameState, depth, alpha, beta, agentIndex = 0):

        value = -sys.maxsize

        for action in gameState.getLegalActions(agentIndex):

            value = max(value, self.value(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta))

            if value > beta: return value
            alpha = max(alpha, value)

        return value

    def min(self, gameState, depth, alpha, beta, agentIndex):
        
        value = sys.maxsize

        for action in gameState.getLegalActions(agentIndex):

            value = min(value, self.value(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta))

            if value < alpha: return value
            beta = min(beta, value)

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.value(gameState.generateSuccessor(0, move), 1) for move in legalMoves]

        bestScore = max(scores)

        return legalMoves[scores.index(bestScore)]

    def value(self, gameState, depth):

        if depth == self.depth * gameState.getNumAgents(): return self.evaluationFunction(gameState)

        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        if depth % gameState.getNumAgents() == 0: return self.max(gameState, depth + 1)
        else: return self.expValue(gameState, depth + 1, depth % gameState.getNumAgents())

    def max(self, gameState, depth, agentIndex = 0):
        
        value = -sys.maxsize

        for action in gameState.getLegalActions(agentIndex):
            value = max(value, self.value(gameState.generateSuccessor(agentIndex, action), depth))

        return value

    def expValue(self, gameState, depth, agentIndex):
        
        value = 0

        for action in gameState.getLegalActions(agentIndex):
            value += self.value(gameState.generateSuccessor(agentIndex, action), depth)

        return value / len(gameState.getLegalActions(agentIndex))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 

    <capsuleWeight>

        The number of remaining capsules times 10. This value ensures that consuming
        a capsule provides more utility than consuming a piece of food.

    </capsuleWeight>

    <foodDist>

        The distance to the closest food from pacman's position.

    </foodDist>

    <return>

        Returning the score of the game state, minus the distance to the closest food, minus the capsule weight
        incentivizes pacman to reduce its distance to the closest food, but consume any capsules it can
        see within its depth of search.

    </return>
    """
    capsuleWeight = len(currentGameState.getCapsules()) * 10
    foodDist = closestFood(currentGameState)

    return currentGameState.getScore() - foodDist - capsuleWeight


"""
Returns the maze length to the nearest food to pacman's
position in the given gamestate
"""
def closestFood(gameState):

    front = util.Queue()
    visited = set([])

    front.push(gameState)

    size = 1
    path = 0

    while not front.isEmpty():

        for i in range(size):

            state = front.pop()
            position = state.getPacmanPosition()

            if gameState.hasFood(position[0], position[1]): return path

            for action in state.getLegalActions():
                successor = state.generatePacmanSuccessor(action)
                if successor.getPacmanPosition() not in visited:
                    front.push(successor)
                    visited.add(successor.getPacmanPosition())

        path += 1
        size = len(front.list)

    return -1

# Abbreviation
better = betterEvaluationFunction