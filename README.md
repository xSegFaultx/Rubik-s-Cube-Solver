# Rubik-s-Cube-Solver
[![Pytorch](https://img.shields.io/badge/%20-Pytorch-grey?style=flat&logo=pytorch)](https://pytorch.org/) \
In this project, I built a Deep Q agent that is able to solve a 3x3 Rubik's Qube from scratch. 

# Environment Setup
Since there's no existing Rubik's cube environment, I need to create the environment myself.

## State
Initially, the color pattern of the cube is considered as the state (3 x 3 x 6 matrix). 
However, the Deep Q network performs really bad using this representation since the Deep Q network cannot fully understand the spatial correlation of a Rubik's cube. 
For example, if the top-right corner of the cube consist of white, red and green, it is very hard for the network to infer that this piece is originally located in the bottom-right corner of the cube. 
Also, since each piece of the cube can have different orientations after the cube is scrambled, this makes it almost impossible for the network to understand the spatial relationships of the cube. 
To solve this problem, I've used the approach introduced in the paper ["SOLVING THE RUBIK’S CUBE WITH APPROXIMATE POLICY ITERATION"](https://openreview.net/pdf?id=Hyfn2jCcKm). 
As shown in the Fig 1.0, each cube has 8 corner pieces and 12 edge pieces. 
The 8 corner pieces can have 24 different location and orientation combinations while the 12 edge pieces can have 20. 
If we one-hot encode the location and the orientation of the corner and the edge pieces, we can get a 20 x 24 tensor. 
This tensor contains spatial information of all the moving pieces of the cube and therefore, it is easier for the network to learn the state of the cube.
<figure>
<img src="https://github.com/xSegFaultx/Rubik-s-Cube-Solver/raw/main/images/fig1.0.PNG" alt="State of The Cube">
<figcaption align = "center"><b>Fig.1.0 - State of The Cube </b></figcaption>
</figure>

## Actions
As shown in Fig 1.1 below, there are in total 12 different actions that can be performed on a Rubik's cube. 
Therefore, these 12 actions are the actions that the agent could perform.
<figure>
<img src="https://github.com/xSegFaultx/Rubik-s-Cube-Solver/raw/main/images/fig1.1.PNG" alt="Actions">
<figcaption align = "center"><b>Fig.1.1 - Actions </b></figcaption>
</figure>

## Reward
Designing the reward function is quite tricky. 
Initially, I would like to award an action that moves the cube closer to the solved state. 
If an action moves the cube 3 steps closer to the solved state, then the reward will be 3. 
However, it turns out that the agent could cheat to get a very high reward. 
For example, if an R move yields a reward of 3 and a Ri move yields a reward of-1, the agent could repeat these two actions to achieve a total reward that is way higher than the total reward of solving the cube. 
To solve this issue, I changed the reward rule so that any action that land the cube on an unsolved state gets a reward of -1 and any actions that land the cube on a solved state gets a reward of 1. 
In order to achieve the highest reward, the agent need to find the shortest path from the given state to the solved state.

## Terminal State
If the cube is solved, then the agent reaches a terminal state. 
Also, the max number of actions an agent can perform in an episode is 50. After 50 actions, the agent will also reach a terminal state.

# Deep Q Learning
Since the Rubik's cube has too many states to store in the memory, I believe it is a good idea to let the Q network predict the Q value of a state-action pair so that no state value table is required. 
Therefore, I chose to use the Deep Q Learning algorithm for this project. 
Since I used a very standard action-replay strategy, I don't think it is necessary to elaborate on how it works or how the Deep Q Learning algorithm work. 
Therefore, in this section, I will focus on the challenges that I encountered in this project.

Challenges:
1. As mentioned in the "State" section, I initially used the color pattern of the cube to represent the state of the cube. For this representation, many different Q networks designs were tried to extract features from the color pattern but all these designs failed. These designs include different 2D CNNs, a 3D CNN, few different DNNs (flattened input). None of these shows sign of convergence and many of them diverges after 1 epoch of training. I've tried to tune the learning rate, batch sizes, and the network structure but none of these helps. Therefore, I start to believe that the representation of the cube may be the issue.After the representation is changed, the problem is solved.
2. Initially, I would like the agent to learn from professional human players. Therefore, I used the [TwophaseSolver](https://github.com/hkociemba/RubiksCube-TwophaseSolver) to calculate how far the current state is from the solved state. However, as mentioned in the "Reward" section, the agent could easily cheat the system. Therefore, I switched to the approach which let the agent solve the cube by reversing the scrambling steps. For example, if the scrambling step is "RRU" then the agent should perform "URR" to solve the cube. There are two advantages of using this approach. First, I could use the reward function mentioned in the "Reward" section which prevent the agent from cheating. Second, I could handcraft the Q values which could be used in the initial training of the model as mentioned in challenge 3.
3. Experience replay usually works very well in many reinforcement learning problems. However,in this project, the state space is very large and the experience replay strategy does not seem to perform very well. When the cube is scrambled for just 1 step, the experience replay strategy works fine. But if the cube is scrambled for 2 steps, the experience replay strategy stops working. This is because most of the states stored in the memory are unsolved states.Initially, only the solved state has a deterministic state value which is 0. The ground truth Q values for other states are predicted by the Q network. If the state space is small (e.g cube is scrambled once), the solved state value 0 can propagate to other unsolved states. However, when the state space increases, it is very hard for the state value 0 to propagate. To solve this problem, I trained a Q network on handcrafted Q values. The Q values are calculated as follow:
If a cube is scrambled 5 times, then its state value is -5. Among all the actions, there must be exactly one action that brings the cube one step closer to the solved state, this action receives a Q value of -4, and all the other actions receive a Q value of -6. This Q value calculation is based on how the reward is given and should represent the true Q value of each state-action
pair (when gamma equals 1). 
I've trained the network on all the possible states that are up to 5 scramble steps away from the solved cube. The training loss, validation loss, training accuracy, and validation accuracy are shown in Fig 3.0 below. After the Q network is trained, I plugged it back to the experience replay framework, and it performs very well.
<figure>
<img src="https://github.com/xSegFaultx/Rubik-s-Cube-Solver/raw/main/images/fig3.0.PNG" alt="Training & Validation Loss">
<figcaption align = "center"><b>Fig.3.0 - Training & Validation Loss </b></figcaption>
</figure>

4. The network trained very fast while using the handcrafted Q value. However, it trained rather slowly in experience replay. It turned out that in the original experience reply framework, only one ground truth Q value (the Q value for the action that the agent actually takes) is calculated for each input. To speed up the training, I calculated all 12 Q values for each input so that the loss for the network is more accurate and thus the network converges faster.

## Training
The Q network is trained on hand-crafted Q values for 800 epochs and takes around 10 minutes on a high-end GPU. 
Then, it is trained on the experience reply framework for another 2000 episodes on cubes that are scramble for 5 and 6 steps and this also takes around 10 minutes on a high-end GPU. 
The experience replay is run 5 times and the total reward of the agent is shown in Fig 4.0 below. 
It can be clearly seen the convergence of the agent based on the mean rewards.
<figure>
<img src="https://github.com/xSegFaultx/Rubik-s-Cube-Solver/raw/main/images/fig4.0.PNG" alt="Average Total Reward">
<figcaption align = "center"><b>Fig.4.0 - Average Total Reward </b></figcaption>
</figure>

# Result
The Deep Q agent is able to solve all the cubes that are scrambled for 4 steps or below. It is able to solve 246924 out of248832 (around 99.23%) cubes that are scrambled 5 steps. Since the 6-step scramble contains too many cubes, the Deep QLearning algorithm is not tested on 6-step scrambled cubes. However, it was tested on a dataset that contains 4000 cubes that are scrambled from 7 to 10 steps (1000 random unique cube for each scramble). It is able to solve 3238 out of 6000 cubes (80.95%) which shows its ability to
generalize on unseen data. The 80.95% accuracy is higher than I expected. I thought the agent would be able to solve cubes that are 1 or 2 steps away from what it is trained with. But the agent performs very well on cubes that are even 4 more scramble steps than the cube in the training set. The detailed result table is shown below.

| Scrambles | Number of Cubes | Solved Percentage |
|:---------:|:---------------:|------------------:|
|     1     |       12        |              100% |
|     2     |       144       |              100% |
|     3     |      1728       |              100% |
|     4     |      20736      |              100% |
|     5     |     248832      |            99.23% |
|   7-10    |      4000       |            80.95% |

However, the agent is not perfect. A perfect cube solver should be able to solve any given cube. There are a few things that I think I could improve in the future.
1. Find a reward function or another training method that allows the agent to learn from professional players since professional players could solve any given cubes.
2. Combining the Deep Q network with MCTS to improve the performance of the agent.
