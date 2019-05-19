# Reinforcement Learning Experiments

Deep reinforcement learning is an exciting area that has had lots of success (e.g., AlphaGo) but getting started in deep reinforcement learning can be a bit of a challenge. Implementing the algorithms requires a bit more work than the standard supervised learning problems and replicating existing benchmarks has its own set of issues--see [Henderson, et al.](https://arxiv.org/abs/1709.06560) or [Amid Fish's blog post](http://amid.fish/reproducing-deep-rl) for a great overview . To get up to speed in this area, the best way is to learn by doing: implement some fundamental algorithms and create a productive environment that allows for quickly testing new ideas. 

The two ideas to start with are double Q-networks (DDQN) and actor-critic algorithms. DDQN were introduced by [van Hasselt, et al.](https://arxiv.org/abs/1509.06461) and, coupled with the memory-replay technique, were able to match human-level performance on a number of Atari games. Actor-critic methods are in another class of algorithms and are among the earliest reinforcement learning methods investigated. A modern deep learning implementation often referenced is [Mnih, et al.'s A3C](https://arxiv.org/abs/1602.01783), but the implementation in this repo is a simpler but no-less-powerful method called Advantage Actor-Critic (A2C). Most of the state-of-the-art methods are variants of these core ideas so DDQN and A2C are natural starting points.

Furthermore, in order to have a functional working environment to easily modify ideas and try out new research directions, I tried to develop modular code and follow OpenAI's principles for [''Spinning Up as a Deep RL Researcher''](https://spinningup.openai.com/en/latest/spinningup/spinningup.html). In particular, I focus on
1. Learn by doing
2. Simplicity is critical
3. Measure everything

To build this repo, I relied heavily on Sutton and Barto's ''Reinforcement Learning'' as well as the papers above. I also found Geron's ''Hand's-On ML'' chapter on reinforcement learning very helpful and a number of online resources, such as [Simonini's posts]( https://medium.freecodecamp.org/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d) and [this GitHub of RLCode](https://github.com/rlcode/reinforcement-learning). 

Finally, please note that this is a work in progress that will be updated as time allows--don't expect perfect outcomes yet!

### What's Here?
**Main takeaway**: benchmarks of fundamental deep RL algorithms in easy-to-use and modify code.

Below there is a brief introduction to general reinforcement learning.

In ./ddqn/ there is an overview of Q-learning and DDQN, files for a bare-bones implementation, and then adaptations of those files to CartPole, LunarLander, and Breakout.

Similarly, ./a2c/ contains an overview of policy-based learning and A2C, files for a bare-bones implementation, and adaptations to CartPole and LunarLander. More applications and benchmarks forthcoming!

## What is RL?
Suppose you're trying to learn to cook an egg. You start out with some basic idea of how this should go and you give it a shot. If it turns out well the first time, great! More likely though is the first attempt doesn't go so well and we need to modify it depending on how it tastes or looks: too salty? overcooked? So you modify your approach and go back and try again: not salty enough? undercooked? If so, you change the recipe and try again until you can cook the best egg possible (or at least one you are satisfied with). 

This is the basic idea of reinforcement learning: figure out what to do so as to maximize reward. However, you aren't told what actions to take; you must discover which actions will give the best reward by trying them. A reinforcement learning algorithm is one that tries to find the best actions (or policy) by interacting with the environment and observing the reward. This basic idea has endless applications: playing Go, learning to walk, financial investing, etc. 

Two big difficulties of this approach are credit attribution and the explore/exploit trade-off. Credit attribution refers to trying to figure out the value of any single decision. Often, the rewards you get from the environment are delayed: you don't taste the egg you cooked until the end so it can be hard to figure out which action in the cooking processes should be changed. Or consider playing Go: the reward is either win or loss but it takes often more than 200 moves to observe this reward. How can we figure out whether any single move was good or bad?

The explore/exploit trade-off is a second theme running through the literature. Going back to the egg-cooking example, suppose it's the case that the first egg you cooked turned out to be a great soft-boiled egg. Based on this, you would want to repeat this process and not make too drastic a change. However, there are many other ways to cook an egg that you won't explore--fried, scrambled, poached, etc. This is the explore/exploit trade-off: exploring is trying new actions and observing the reward; exploiting is repeating the known good actions. It's not possible to do both at the same time, so there needs to be a way of to balance between the alternatives: perfecting the soft-boiled egg but also trying to cook a fried egg.

In the directories above, you'll find two different approaches for addressing these issues.
