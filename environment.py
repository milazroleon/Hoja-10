class BanditProblem:
    
    def __init__(self, distributions):
        self.distributions = distributions
        self.num_arms = len(distributions)

        self.means = [np.dot(values, probs) for values, probs in distributions]
        self.optimal_arm = np.argmax(self.means)
        self.optimal_mean = self.means[self.optimal_arm]

    def pull(self, arm):
        values, probs = self.distributions[arm]
        return np.random.choice(values, p=probs)
        
    def simulate_policy(self, policy, max_t):
        policy.setup(self.num_arms)
        total_reward = 0
        optimal_actions = 0
        cumulative_regret = 0
        data = []

        for t in range(1, max_t + 1):
            arm = policy.choose()
            reward = self.pull(arm)
            policy.tell_reward(arm, reward)

            total_reward += reward
            if arm == self.optimal_arm:
                optimal_actions += 1
            cumulative_regret += self.optimal_mean - self.means[arm]

            avg_reward = total_reward / t
            optimal_rate = optimal_actions / t

            data.append([arm, reward, avg_reward, optimal_rate, cumulative_regret])

        df = pd.DataFrame(data, columns=["arm", "reward", "avg_reward", "optimal_rate", "cum_regret"])
        return df
