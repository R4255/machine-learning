#markov models we deal with probability distributions
#example to predict the weather forecast given the probability of the different event
#hidden markov model is a finit set of states each of which is associated with a probabilitiy distribution
#we actually dont care about the states we care about the observations from each state
#we define how many state we have but we dont care what that state is 
#each of the state has a particular outcome or observation associated with it based on probability distribution 
#states transition observation ,we dont just look at the states , we just have to know how many we have and the transition and
#the observation probability in each of them 
#this hidden markov model will be used to predict the future events based on the past ones 

import tensorflow_probability as tfp
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# cold days are encoded by 0 and hot days by 1
# the first day in our sequence has an 80% chance of being cold
# a cold day has a 30% chance of being followed by a hot day
# a hot day has a 20% chance of being followed by a cold day

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# the loc represents the mean and scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7  # num_steps mainly define us for how many days do we want it
)
mean = model.mean()

# due to the way TensorFlow works on the lower level, we need to evaluate part of the graph
# from within a session to see the value of this tensor
# in the new version of TensorFlow, we need to use tf.compat.v1.Session() rather than just tf.Session()

# Use this if using TensorFlow 2.x and eager execution
print(mean.numpy())

# Use this if using TensorFlow 1.x
#with tf.Session() as sess:
 #   print(sess.run(mean))
