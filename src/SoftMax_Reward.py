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

def soft_max(temperature, arms_num, arm_mean):

    # We calculate here for each arm his value with the exponentiel formula
    softMax_list = []
    # We need the sum of all the exponentiel value for the formula
    sum_mean_with_exp = 0;
    for i in range(0, arms_num):
      sum_mean_with_exp += np.exp(temperature*arm_mean[i])

    for i in range(0, arms_num):
      softMax_list.append(np.exp(temperature*arm_mean[i])/sum_mean_with_exp)

    # Here we chose the arm randomly but with an associated probability for each arm (probability = value calculated before)
    chosen_arm = np.random.choice(range(arms_num), p=softMax_list)
    return chosen_arm

# Machine learning
def machine_learning(number_of_arms, number_of_bandit_pb, num_of_steps, temperature):

    # List with 2000 lists of the 1000 value that were earn
    res = []

    for problem_number in range(0, number_of_bandit_pb):

        # List with reward earn each step (1000 values)
        earned_reward = []
        # Initialization of the arm arm_center_gaussian value for the center of the gaussian distribution
        arm_center_gaussian = np.random.normal(size = number_of_arms)
        # List with for each arm his current reward meaning
        arm_mean = [0.0]*number_of_arms
        # List to compt how many times each arm was chosen
        occurence = [0]*number_of_arms

        for step in range(0, num_of_steps):

            # Selected arm with soft_max_greedy function
            chosen_arm = soft_max(temperature, number_of_arms, arm_mean)
            # Reward is the random value following gaussian distribution with the center of the chosen_arm
            reward = np.random.normal(arm_center_gaussian[chosen_arm])
            occurence[chosen_arm] += 1
            # Actualization of the reward mean for the chosen arm
            arm_mean[chosen_arm] = ((arm_mean[chosen_arm] * occurence[chosen_arm] + reward) / (occurence[chosen_arm]+1))
            earned_reward.append(reward)

        # Add to returned list the reward for this bandit problem
        res.append(earned_reward)
        advance_statut(float(problem_number) / float(number_of_bandit_pb-1) * 100.0)
    return res

# Graphic Display
def graphic_display(final_values_soft_max1, final_values_soft_max2, final_values_soft_max3, my_time):
    pyplot.plot(range(0, len(final_values_soft_max1[0])), np.mean(final_values_soft_max1, axis=0), "b-", linewidth = 2, label = "temperature = 2")
    pyplot.plot(range(0, len(final_values_soft_max2[0])), np.mean(final_values_soft_max2, axis=0), "r-", linewidth = 2, label = "temperature = 6")
    pyplot.plot(range(0, len(final_values_soft_max3[0])), np.mean(final_values_soft_max3, axis=0), "g-", linewidth = 2, label = "temperature = 0")
    pyplot.legend(loc = "upper left")
    pyplot.xlabel("Steps")
    pyplot.ylabel("Average reward")
    pyplot.title("Learning with Soft_maxs")
    print("Time = ", time.time() - my_time)
    pyplot.show()

my_time = time.time()
final_values_soft_max1 = machine_learning(10, 2000, 1000, 2)
final_values_soft_max2 = machine_learning(10, 2000, 1000, 6)
final_values_soft_max3 = machine_learning(10, 2000, 1000, 0) 

graphic_display(final_values_soft_max1, final_values_soft_max2, final_values_soft_max3, my_time)
