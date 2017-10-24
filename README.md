# Differential-Training-Of-Rollout-Policies
A common reinforcement learning approach to nd optimal policies
is through the estimation of value functions. Bertsekas[1] wrote an in-
teresting paper arguing why it might be better to learn functions that
measure the dierence in value between states, rather than the value of
states. In this exploratory project, we propose to validate this hypothesis
by implementing the dierential training method proposed by Bertsekas
for learning the value dierences between pairs of states on simple rein-
forcement learning environments with nite state and action space. If
possible, we'd also try to implement this method on more complex con-
tinuous state space environments using neural networks as function ap-
proximators.
