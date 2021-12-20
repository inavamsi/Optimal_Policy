# Optimal_Policy
Learn the optimal public policy to end the epidemic using Reinforcement Learning.

### Current Public Policies :

1. Vaccination Block Policy
2. Quarantine Block Policy

### Current variants are :

The RL Agent uses variants of the Deep Q Learning Networks (DQN).

1. Linear DQN (The only network without convolutional networks)
2. DQN (with and without target networks)
3. Double DQN
4. Dueling DQN
5. Dueling Double DQN

### Requirements
* Python3.8+
* Numpy
* PyTorch
* Matplotlib

See requirements.txt

### Usage :

usage:
python main.py [-h] [-gs GRID_SIZE] [-vs VAX_SIZE] [-net NETWORK] [-me MAX_EPD] [-lr LEARNING_RATE] [-g GAMMA] [-bs BATCH_SIZE] [-buf MAX_BUFFER_SIZE] [-emx EPS_MAX] [-emn EPS_MIN]
               [-ed EPS_DEC] [-us UPDATE_STEPS]

optional arguments:
  -h, --help            show this help message and exit

  -gs GRID_SIZE, --grid_size GRID_SIZE
                        Grid size S x S. Default S = 12

  -vs VAX_SIZE, --vax_size VAX_SIZE
                        Vaccination size V x V. Default V = 4

  -net NETWORK, --network NETWORK
                        Network to use. Options = LinearDQN, SimpleConvDQN, ConvDQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN. Default = ConvDQN

  -me MAX_EPD, --max_epd MAX_EPD
                        Max Episodes to run. Default = 10000

  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning Rate. Default = 3e-2

  -g GAMMA, --gamma GAMMA
                        Gamma value. Default = 0.95

  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size. Default = 16

  -buf MAX_BUFFER_SIZE, --buffer MAX_BUFFER_SIZE
                        Max buffer size. Default = 10000

  -emx EPS_MAX, --eps_max EPS_MAX
                        Maximum Epsilon. Default = 1.0

  -emn EPS_MIN, --eps_min EPS_MIN
                        Minimum Epsilon. Default = 0.01

  -ed EPS_DEC, --eps_dec EPS_DEC
                        Epsilon Decrement. Default = 0.0001

  -us UPDATE_STEPS, --update_steps UPDATE_STEPS
                        Target Network update steps for ConvDQN. Default = 200
                        
