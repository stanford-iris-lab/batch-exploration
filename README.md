# Batch Exploration with Examples for Scalable Robotic Reinforcement Learning

This code implements the following paper: 

> [Batch Exploration with Examples for Scalable Robotic Reinforcement Learning](https://sites.google.com/view/batch-exploration). 
>
> Annie S. Chen*, Hyunji Nam*, Suraj Nair*, Chelsea Finn. [arXiv preprint](https://arxiv.org/abs/2010.11917), 2020.

## Abstract
Learning from diverse offline datasets is a promising path towards learning general purpose robotic agents. However, a core challenge in this paradigm lies in collecting large amounts of meaningful data, while not depending on a human in the loop for data collection. One way to address this challenge is through task-agnostic exploration, where an agent attempts to explore without a task-specific reward function, and collect data that can be useful for any downstream task. While these approaches have shown some promise in simple domains, they often struggle to explore the relevant regions of the state space in more challenging settings, such as vision based robotic manipulation. This challenge stems from an objective that encourages exploring everything in a potentially vast state space. To mitigate this challenge, we propose to focus exploration on the important parts of the state space using weak human supervision. Concretely, we propose an exploration technique, Batch Exploration with Examples (BEE), that explores relevant regions of the state-space, guided by a modest number of human provided images of important states. These human provided images only need to be collected once at the beginning of data collection and can be collected in a matter of minutes, allowing us to scalably collect diverse datasets, which can then be combined with any batch RL algorithm. We find that BEE is able to tackle challenging vision-based manipulation tasks both in simulation and on a real Franka robot, and observe that compared to task-agnostic and weakly-supervised exploration techniques, it (1) interacts more than twice as often with relevant objects, and (2) improves downstream task performance when used in conjunction with offline RL.

## Installation
1. Clone the repository by running:
```
git clone https://github.com/nam630/batch_exploration.git
cd batch_exploration
```
2. Install Mujoco 2.0 and mujoco-py. Instructions for this are [here](https://github.com/openai/mujoco-py#install-mujoco).

3. Create and activate conda environment with the required prerequisites:
```
conda env create -f conda_env.yml
conda activate batch_exp
```

4. Our simulation env depends on Meta-World. Install it [here](https://github.com/tianheyu927/metaworld).

5. Install the simulation env by running:
```
cd envs
pip install -e .
```

## Batch Exploration Phase: Online data collection

Our simulation environment supports the following domains: a block env with 3 blocks, a door env with 3 tall distractor towers, a door env with 5 tall distractor towers, and drawer env with 6 distractor blocks. To run with each of these as the target, add one of the following flags respectively: ```--goal_block $BLOCK_NUM, --door $NUM_DISTRACTORS, --drawer```, where ```$BLOCK_NUM``` ranges from 0-2 based on the target block (green, pink, or blue) and ```$NUM_DISTRACTORS``` is either 3 or 5. All default args are listed in [here](https://github.com/nam630/batch_exploration/blob/master/configs/args.py).

### 1. Commands for BEE:
Use the flag ```--use_classifiers max```.
```
python run.py --date "[dirname]" --door 5 --use_classifiers "max" --max_epoch 2101 --seed 0   # BEE door env w/ 5 distractors
python run.py --date "[dirname]" --door 3 --use_classifiers "max" --max_epoch 2101 --seed 0   # BEE door env w/ 3 distractors
```

### 2. Commands for Model Disagreement:
Use the flag ```--dynamics_var```.
```
python run.py --date "[dirname]" --drawer --dynamics_var  # Disagreement w/ the drawer env
```

### 3. Commands for State Marginal Matching (SMM):
Use the flag ```--smm```.
```
python run.py --date "[dirname]" --smm --goal_block 0   # SMM w/ green block as goal
python run.py --date "[dirname]" --smm --goal_block 0   # SMM w/ pink block as goal
python run.py --date "[dirname]" --smm --goal_block 0   # SMM w/ blue block as goal
```

## Batch RL Phase: Downstream Planning

For downstream batch RL using the collected data, we train a visual dynamics model using Stochastic Variational Video Prediction (SV2P). The code base can be found [here](https://github.com/tensorflow/tensor2tensor). 

To run planning with a trained SV2P model, call ```python sv2p_plan.py```. The args should be changed in the file [here](https://github.com/nam630/batch_exploration/blob/master/sv2p_plan.py) to point to the correct directories where the SV2P model is saved.


