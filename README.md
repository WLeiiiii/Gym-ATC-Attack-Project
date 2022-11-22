# Explainable and Safe Reinforcement Learning for Autonomous Air Mobility

## Introduction
It is the source code of **"Explainable and Safe Reinforcement Learning for Autonomous Air Mobility"**, which presents a novel deep reinforcement learning (DRL)
controller to aid conflict resolution for autonomous free flight. To study the safety under adversarial attacks, we additionally propose an adversarial attack strategy that can impose both safety-oriented and efficiency-oriented attacks.

## Requirements

* Python 3.8
* [PyTorch](http://pytorch.org/)

In order to install requirements, follow:

```bash
pip install -r requirements.txt
```
## Details

### Environment

The definition of environment is in `envs`: 

`envs/SimpleATC_env` is for **traditional DQN** agent with global perception;

`envs/SimpleATC_env_v2` is for **safety-aware DQN(SafeDQN)** agent with global perception;

`envs/SimpleATC_env_flexible` is for **traditional DQN** agent with local perception;

`envs/SimpleATC_env_flexible_v2` is for **safety-aware DQN(SafeDQN)** agent with local perception.



Parameter of the environments can be found in `envs/Config.py`

### DQN Agents

### Adversarial Attacks

## Demo
|                  |         Without Attack         |                     Uniform Attack                      |           Strategically-Timed Attack            |
|:----------------:|:------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------:|
|    **DQN-10**    |   ![image](gifs/DQN-10.gif)    |           ![image](gifs/DQN-10-UniAttack.gif)           |       ![image](gifs/DQN-10-STAttack.gif)        |
|   **DQN-X10**    |   ![image](gifs/DQN-X10.gif)   |          ![image](gifs/DQN-X10-UniAttack.gif)           |       ![image](gifs/DQN-X10-STAttack.gif)       |
|  **safeDQN-10**  | ![image](gifs/safeDQN-10.gif)  |         ![image](gifs/safeDQN-10-UniAttack.gif)         |     ![image](gifs/safeDQN-10-STAttack.gif)      | 
| **safeDQN-X10**  | ![image](gifs/safeDQN-X10.gif) |        ![image](gifs/safeDQN-X10-UniAttack.gif)         |     ![image](gifs/safeDQN-X10-STAttack.gif)     |






