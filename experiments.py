import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from policy import EpsilonGreedyPolicy, UCB
from environment import BanditProblem

def run_experiment(bandit_problem, T=1000, R=100, init_mean_value=0):
    policies = {
        "ε-Greedy (ε=0.1)": lambda: EpsilonGreedyPolicy(0.1, init_mean_value),
        "ε-Greedy (ε=1/t)": lambda: EpsilonGreedyPolicy(lambda t: 1 / t, init_mean_value),
        "UCB1": lambda: UCB(c=2, init_mean_value=init_mean_value)
    }

    results = {name: [] for name in policies}

    for name, make_policy in policies.items():
        for _ in range(R):
            policy = make_policy()
            env = BanditProblem(bandit_problem)
            df = env.simulate_policy(policy, T)
            results[name].append(df)

    metrics = ["avg_reward", "optimal_rate", "cum_regret"]
    avg_results = {metric: {} for metric in metrics}

    for metric in metrics:
        for name, runs in results.items():
            avg_df = pd.concat([df[metric] for df in runs], axis=1).mean(axis=1)
            avg_results[metric][name] = avg_df

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, curve in avg_results[metric].items():
            plt.plot(curve, label=name)
        plt.title(metric.replace("_", " ").capitalize())
        plt.xlabel("Timestep")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":
    arms_3 = [([1, 2], [0.5, 0.5]), ([3, 0], [0.2, 0.8]), ([5, 0], [0.1, 0.9])]
    run_experiment(arms_3, T=500, R=10)
