import numpy as np 


class Policy:

    def __init__(self, init_mean_value=0):
        self.init_mean_value = init_mean_value
        self.num_arms = None
        self.counts = None
        self.values = None
        self.t = 0
    
    def setup(self, num_arms):
        """
            sets up the policy
        """
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.ones(num_arms) * self.init_mean_value
        self.t = 0
        
    def choose(self) -> int:
        """
            returns the index of the next action the agent wants to take
        """
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    @property
    def mean_estimates(self):
        return self.values
    
class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, init_mean_value=0):
        super().__init__(init_mean_value)
        self.epsilon = epsilon
    
    def setup(self, num_arms):
        super().setup(num_arms)

    def choose(self) -> int:
        self.t += 1
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.values)
    
    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)

class UCB(Policy):

    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value)
        self.c = c

    def setup(self, num_arms):
        super().setup(num_arms)

    @property
    def exploration_terms(self):
        """
            must return a numpy array with the exploration term for each arm (exploration constant c not included)
        """
        exploration = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self.counts[arm] > 0:
                exploration[arm] = np.sqrt(2 * np.log(self.t + 1) / self.counts[arm])
            else:
                exploration[arm] = np.inf
        return exploration

    def choose(self):
        self.t += 1
        ucb_values = self.values + self.c * self.exploration_terms
        return np.argmax(ucb_values)
    
    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
