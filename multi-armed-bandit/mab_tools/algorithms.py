import math
import numpy as np
import random


def ind_max(x: list) -> int:
    m = max(x)
    return x.index(m)


class EpsilonGreedy():
    def __init__(self, epsilon: float, counts: int, values: float) -> None:
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        return 

    def initialize(self, n_arms: int) -> None:
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return 
    
    def select_arm(self) -> int:
        if random.random() > self.epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))
    
    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        
        value = self.values[chosen_arm]
        new_value = (value * (n-1)/n) + reward / n
        self.values[chosen_arm] = new_value
        return 
    
def categorical_draw(probs: list) -> int:
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i
        
    return len(probs) - 1


class SoftMax():
    def __init__(self, temperature: float, counts: int, values: float) -> None:
        self.temperature = temperature
        self.counts = counts
        self.values = values
        
    def initialize(self, n_arms: int) -> None:
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    def select_arm(self) -> int:
        z = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) / z for v in self.values]
        return categorical_draw(probs)
    
    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        
        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value
        return
    

class UCB1():
    def __init__(self, counts: int, values: float) -> None:
        self.counts = counts
        self.values = values
    
    def initialize(self, n_arms: int) -> None:
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    def select_arm(self) -> int:
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        
        ucb_values = [0.0 for col in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt(2 * math.log(total_counts) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return ind_max(ucb_values)
    
    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = sum(self.counts)
         
        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value
        return
    
    
class TompsonSampling():
    def __init__(self, counts_alpha: float, counts_beta: float, values: float) -> None:
        self.counts_alpha = counts_alpha
        self.counts_beta = counts_beta
        self.alpha = 1
        self.beta = 1
        self.values = values
    
    def initialize(self, n_arms: int) -> None:
        self.counts_alpha = [0.0 for col in range(n_arms)]
        self.counts_beta = [0.0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        
    def select_arm(self) -> int:
        theta = [random.betavariate(self.counts_alpha[arm] + self.alpha,
                                    self.counts_beta[arm] + self.beta)
                 for arm in range(len(self.counts_alpha))]
        return ind_max(theta)
    
    def update(self, chosen_arm: int, reward: float) -> None:
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1
        
        n = sum(self.counts_alpha) + sum(self.counts_beta)
         
        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value
        return