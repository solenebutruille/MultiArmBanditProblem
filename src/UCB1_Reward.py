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




def machine_learning(number_of_arms, number_of_bandit_pb, num_of_steps):

    # List with 2000 lists of the 1000 value that were earn
    res = []

    for problem_number in range(0, number_of_bandit_pb):

        # List with reward earn each step (1000 values)
        earned_reward = []
        # List who will contain for each arm, its 95% confidence intervalle
        value_95 = [0.0]*number_of_arms
        # Initialization of the arm arm_center_gaussian value for the center of the gaussian distribution
        arm_center_gaussian = np.random.normal(size = number_of_arms)
        # List with for each arm his current reward meaning
        arm_mean = [0.0]*number_of_arms
        # List to compt how many times each arm was chosen
        occurence = [1]*number_of_arms

        for step in range(0, num_of_steps):

            # Chosen_arm is the one with the best chance of having a good reward regardin to the 95% confidence intervalle
            chosen_arm = np.argmax(value_95)
            reward = np.random.normal(arm_center_gaussian[chosen_arm])
            arm_mean[chosen_arm] = ((arm_mean[chosen_arm] * (occurence[chosen_arm]-1) + reward) / (occurence[chosen_arm]))
            occurence[chosen_arm] += 1
            earned_reward.append(reward)
            # Here we calculate for each arm 95% confidence intervalle
            for i in range(0, number_of_arms) :
              value_95[i] = (arm_mean[i] + np.sqrt(2*np.log(step+1)/occurence[i]))

        # Add to returned list the reward for this bandit problem
        res.append(earned_reward)
        advance_statut(float(problem_number) / float(number_of_bandit_pb) * 100.0)

    return res

# Graphic Display
def graphic_display(final_values_UCB1, my_time):
    pyplot.plot(range(0, len(final_values_UCB1[0])), np.mean(final_values_UCB1, axis=0), "b-", linewidth=2)
    pyplot.xlabel("Steps")
    pyplot.ylabel("Average Reward")
    pyplot.title("Learning with UCB1")
    print("Time = ", time.time() - my_time)
    pyplot.show()

my_time = time.time()
#number_of_arms, number_of_bandit_pb, num_of_steps
final_values_UCB1 = machine_learning(10, 2000, 1000)

graphic_display(final_values_UCB1, my_time)
