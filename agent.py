import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = 0.15
        self.epsilon = 0.0001
        self.gamma = 0.95
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        values = self.Q[state]

        probs = np.ones_like(values) * (self.epsilon/self.nA)  
        probs[np.argmax(values)] = 1-self.epsilon + self.epsilon/self.nA
        return np.random.choice(np.arange(self.nA), p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """       
        st=state
        at=action
        st_1=next_state

        if done:
            self.Q[st][at] += self.alpha*(reward - self.Q[st][at])            

        else:
            self.Q[st][at] += self.alpha*( reward \
                + self.gamma*(self.Q[st_1].max())-self.Q[st][at] )

        #self.Q[state][action] += 1