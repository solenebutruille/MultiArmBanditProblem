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

# Machine learning
def machine_learning(number_of_arms, number_of_bandit_pb, delta, epsilon):

  # res is what we will return with meaning value on all problems
  res = []

  for problem_number in range(0, number_of_bandit_pb) :
    mean_value_step = []

    # At the beginning, we have all arms because none was eliminated
    restrained_arm = []
    for arm in range(0, number_of_arms) :
      restrained_arm.append(arm)

    # Initialization of the arm arm_center_gaussian value for the center of the gaussian distribution
    arm_center_gaussian = np.random.normal(size = number_of_arms)

    # Number of iterations we will do depending on delta and epsilon_greedy
    number_of_iteration = (2/(epsilon*epsilon))*np.log(3/delta)
    number_of_iteration = int(round(number_of_iteration) + 1)

    # Variable stocking how many arm we still have
    nb_arm = len(restrained_arm)

    while nb_arm != 1 :

        # New value of the arm number
        nb_arm = len(restrained_arm)
        # Mean value on the average for each arm to choose which arm we keep at the end
        mean_value = [0.0]*nb_arm

        for arm in range(0, nb_arm):
            sum = 0
            for i in range(0, number_of_iteration) :
                reward = np.random.normal(arm_center_gaussian[restrained_arm[arm]])
                sum += reward
                mean_value_step.append(np.mean(reward))
            mean_value[arm] = sum / number_of_iteration


        sort(mean_value)
        restrained_arm_new = []
      	quarter_value = 0
      	if((int)(nb_arm%4) != (nb_arm%4)) :
      		quarter_value = ((int)(nb_arm%4)+1)
      	else :
      		quarter_value = (nb_arm%4)
        for arm in range(quarter_value, nb_arm) :
            restrained_arm_new.append(restrained_arm[arm])

        restrained_arm = restrained_arm_new
    #    mean_value_step.append(np.mean(mean_value))
    advance_statut(float(problem_number) / float(number_of_bandit_pb-1) * 100.0)
    res.append(mean_value_step)

  return res

# Graphic Display
#def graphic_display(mean_value_step1, mean_value_step2, mean_value_step3,  my_time) :
def graphic_display(mean_value_step1, my_time) :
    pyplot.plot(range(0, len(mean_value_step1[0])), np.mean(mean_value_step1, axis=0), "b-", linewidth = 2, label = "Epsilon=Delta=0.3")
#    pyplot.plot(range(0, len(mean_value_step2[0])), np.mean(mean_value_step2, axis=0), "r-", linewidth = 2, label = "Epsilon=Delta=0.5")
#    pyplot.plot(range(0, len(mean_value_step3[0])), np.mean(mean_value_step3, axis=0), "g-", linewidth = 2, label = "Epsilon=Delta=1")
 #   pyplot.plot(range(0, len(mean_value_step4[0])), np.mean(mean_value_step4, axis=0), "d-", linewidth = 2, label = "Epsilon=Delta=0.1")
    pyplot.legend(loc = "lower right")
    pyplot.xlabel("Steps")
    pyplot.ylabel("Average Reward")
    pyplot.title("Learning with MED")
    print("Time = ", time.time() - my_time)
    pyplot.show()

my_time = time.time()
#final_values1 = machine_learning(10, 2000, 0.3, 0.3)
#final_values2 = machine_learning(10, 2000, 0.5, 0.5)
#final_values3 = machine_learning(10, 2000, 1, 1)
final_values4 = machine_learning(10, 2000, 0.1, 0.1)

#graphic_display(final_values1, final_values2, final_values3,  my_time)
