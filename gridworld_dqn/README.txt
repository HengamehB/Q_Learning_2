This is a simple test of DQN technology in a "gridworld" game,
see https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df.
In each game, a random NxN board (default N = 5) is generated
with one player (blue), guard (red), and target (green). The AI
moves in cardinal directions attempting to reach the target,
avoiding the guard. The guard is given a probability p to move
in a random direction.

For a random board the game is not winnable with 100% probability,
but smart play can get quite close.

