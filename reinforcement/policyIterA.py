import util

class PolicyIterationAgent:

    def __init__(self, mdp, gamma=0.9, iterations=100):
        self.mdp = mdp
        self.gamma = gamma
        self.iterations = iterations
        self.values = util.Counter()
        self.policy = util.Counter()
        self.initValues()

    def initValues(self):
        for state in self.mdp.getStates():
            self.values[state] = 0

    def getValue(self, state):
        return self.values[state]

    def getQValue(self, state, action):
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob * (self.mdp.getReward(state, action, next_state) + self.gamma * self.get_value(next_state))
        return q_value

    # def getBestAction(self, state):
    #     best_action = None
    #     best_q_value = float('-inf')
    #     for action in self.mdp.getPossibleActions(state):
    #         q_value = self.getQValue(state, action)
    #         if q_value > best_q_value:
    #             best_q_value = q_value
    #             best_action = action
    #     return best_action

    def getBestAction(self, state):
        legal_actions = self.mdp.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        best_action = None
        best_value = float('-inf')
        for action in legal_actions:
            value = 0
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                reward = self.mdp.getReward(state, action, next_state)
                value += prob * (reward + self.discount * self.values[next_state])
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.policy[state]

    def getActions(self, state):
        return self.mdp.getPossibleActions(state)

    def updateValues(self):
        for i in range(self.iterations):
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    action = self.getPolicy(state)
                    self.values[state] = self.getQValue(state, action)

    def updatePolicy(self):
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                best_action = self.getBestAction(state)
                self.policy[state] = best_action

    def init(self):
        self.updateValues()
        self.updatePolicy()

    def getAction(self, state):
        return self.getPolicy(state)