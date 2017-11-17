# Differential-Training-Of-Rollout-Policies
A common reinforcement learning approach to find optimal policies
is through the estimation of value functions. Bertsekas[1] wrote an interesting paper arguing why it might be better to learn functions that
measure the difference in value between states, rather than the value of
states. In this exploratory project, we propose to validate this hypothesis
by implementing the differential training method proposed by Bertsekas
for learning the value differences between pairs of states on simple rein-
forcement learning environments with finite state and action space. If
possible, we'd also try to implement this method on more complex con-
tinuous state space environments using neural networks as function ap-
proximators.

# References
1. Dimitri P Bertsekas. Differential training of rollout policies. 1997.
