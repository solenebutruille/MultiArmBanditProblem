# -*- coding: utf-8 -*-
"""
@author: Butruille Solène

"""
import sys
import numpy as np
import matplotlib.pyplot as pyplot
import time

# Display percentage of advancement
def advance_statut(update):
    sys.stdout.write("\r%d%%" % update)

# Epsilon_greedy algorithm
def epsilon_greedy(epsilon, number_of_arms, arm_mean):
    # We take a random number between 0 and 1 and compare it to epsilon
    if np.random.random_sample() >= epsilon:
        # We take the one with the best average at this time
        return np.argmax(arm_mean)
    else:
        # We choose a random arm
        return np.random.randint(0, number_of_arms)

# Machine learning
def machine_learning(number_of_arms, number_of_bandit_pb, num_of_steps, epsilon):

    # List with 2000 lists of the 1000 value that were earn
    res = []

    for problem_number in range(0, number_of_bandit_pb):

        # List with reward earn each step (1000 values)
        best_value = []
        # Initialization of the arm arm_center_gaussian value for the center of the gaussian distribution
        arm_center_gaussian = np.random.normal(size = number_of_arms)
        # List with for each arm his current reward meaning
        arm_mean = [0.0]*number_of_arms
        # List to compt how many times each arm was chosen
        occurence = [0]*number_of_arms

        for step in range(0, num_of_steps):

            # Selected arm with epsilon_greedy function
            chosen_arm = epsilon_greedy(epsilon, number_of_arms, arm_mean)
            # Reward is the random value following gaussian distribution with the center of the chosen_arm
            reward = np.random.normal(arm_center_gaussian[chosen_arm])
            occurence[chosen_arm] += 1
            # Actualization of the reward mean for the chosen arm
            arm_mean[chosen_arm] = ((arm_mean[chosen_arm] * occurence[chosen_arm] + reward) / (occurence[chosen_arm]+1))

            # We add to the list if it was or not the best value
            if max(arm_center_gaussian) == arm_center_gaussian[chosen_arm] :
              best_value.append(1)
            else :
              best_value.append(0)

        # Add to returned list the reward for this bandit problem
        res.append(best_value)
        advance_statut(float(problem_number) / float(number_of_bandit_pb-1) * 100.0)
    return res

# Graphic Display
def graphic_display(final_values_epsilon1, final_values_epsilon2, final_values_epsilon3, my_time):
    # We print for each step the vzlue we got when we do the meaning on the 2000 bzndit problem
    pyplot.plot(range(0, len(final_values_epsilon1[0])), np.mean(final_values_epsilon1, axis=0)*100, "b-", linewidth=2, label = "Epsilon = 0.1")
    pyplot.plot(range(0, len(final_values_epsilon2[0])), np.mean(final_values_epsilon2, axis=0)*100, "r-", linewidth=2, label = "Epsilon = 0.01")
    pyplot.plot(range(0, len(final_values_epsilon3[0])), np.mean(final_values_epsilon3, axis=0)*100, "g-", linewidth=2, label = "Epsilon = 0")
    pyplot.legend(loc = "upper left")
    pyplot.xlabel("Steps")
    pyplot.ylabel("% Optimal Action")
    pyplot.ylim(0,100)
    print("Time = ", time.time() - my_time)
    pyplot.title("Learning with Epsilon-greedy")
    pyplot.show()

my_time = time.time()
final_values_epsilon1 = machine_learning(10, 2000, 1000, 0.1)
final_values_epsilon2 = machine_learning(10, 2000, 1000, 0.01)
final_values_epsilon3 = machine_learning(10, 2000, 1000, 0)

graphic_display(final_values_epsilon1, final_values_epsilon2, final_values_epsilon3, my_time)
