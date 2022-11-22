# Explainable and Safe Reinforcement Learning for Autonomous Air Mobility

## Introduction
It is the source code of "Explainable and Safe Reinforcement Learning for Autonomous Air Mobility", which presents a novel deep reinforcement learning (DRL)
controller to aid conflict resolution for autonomous free flight. To study the safety under adversarial attacks, we additionally propose an adversarial attack strategy that can impose both safety-oriented and efficiency-oriented attacks.

## Requirements

* Python 3.8
* [PyTorch](http://pytorch.org/)

In order to install requirements, follow:

```bash
pip install -r requirements.txt
```


## Demo
|                  |         Without Attack         |                     Uniform Attack                      |           Strategically-Timed Attack            |
|:----------------:|:------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------:|
|    **DQN-10**    |   ![image](gifs/DQN-10.gif)    |           ![image](gifs/DQN-10-UniAttack.gif)           |       ![image](gifs/DQN-10-STAttack.gif)        |
|   **DQN-X10**    |   ![image](gifs/DQN-X10.gif)   |          ![image](gifs/DQN-X10-UniAttack.gif)           |       ![image](gifs/DQN-X10-STAttack.gif)       |
|  **safeDQN-10**  | ![image](gifs/safeDQN-10.gif)  |         ![image](gifs/safeDQN-10-UniAttack.gif)         |     ![image](gifs/safeDQN-10-STAttack.gif)      | 
| **safeDQN-X10**  | ![image](gifs/safeDQN-X10.gif) |        ![image](gifs/safeDQN-X10-UniAttack.gif)         |     ![image](gifs/safeDQN-X10-STAttack.gif)     |

## Details

### Environment

### DQN Agents

### Adversarial Attacks




