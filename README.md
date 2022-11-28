# Explainable and Safe Reinforcement Learning for Autonomous Air Mobility

## Introduction
It is the source code of **"[Explainable and Safe Reinforcement Learning for Autonomous Air Mobility](https://arxiv.org/pdf/2211.13474.pdf)"**, which presents a novel deep reinforcement learning (DRL)
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

* `envs/SimpleATC_env_global.py` is for **traditional DQN** agent with fixed airways and global perception

* `envs/SimpleATC_env_global_v2.py` is for **safety-aware DQN(SafeDQN)** agent with fixed airways and global perception

* `envs/SimpleATC_env_local.py` is for **traditional DQN** agent with fixed airways and local perception

* `envs/SimpleATC_env_local_v2.py` is for **safety-aware DQN(SafeDQN)** agent with fixed airways and local perception

* `envs/SimpleATC_env_local_x.py` is for **traditional DQN** agent with random airways and local perception

* `envs/SimpleATC_env_local_x_v2.py` is for **safety-aware DQN(SafeDQN)** agent with random airways and local perception

Parameter of the environments can be found in `envs/config.py`. And here you can change the related parameters according to your own needs. 

### DQN Agents

By importing different environments in `envs`, different models can be trained and evaluated in `agents`:

* `agents/dqn_simple_env` is for **traditional DQN** agent
* `agents/dqn_simple_env_v2` is for **safety-aware DQN(SafeDQN)** agent

```bash
# take traditional DQN as an example

# For training:
python dqn_simple_env.py --train=True --save_path=" "

# For evaluating:
python dqn_simple_env.py --load_path=" "
```


You can find the DQN structure in `models/dqn_model`.
### Adversarial Attacks

The adversarial attack methods are in `attacks`:

* `v1` for **traditional DQN** agent
* `v2` for **safety-aware DQN(SafeDQN)** agent

## Demo
Here, we present the demos of four models under
10 routes scenarios without/with adversarial attacks. For safeDQN and safeDQN-X, we only present the results under safety-oriented attacks.

|               |         Without Attack         |                     Uniform Attack                      |           Strategically-Timed Attack            |
|:-------------:|:------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------:|
|    **DQN**    |   ![image](gifs/DQN-10.gif)    |           ![image](gifs/DQN-10-UniAttack.gif)           |       ![image](gifs/DQN-10-STAttack.gif)        |
|   **DQN-X**   |   ![image](gifs/DQN-X10.gif)   |          ![image](gifs/DQN-X10-UniAttack.gif)           |       ![image](gifs/DQN-X10-STAttack.gif)       |
|  **safeDQN**  | ![image](gifs/safeDQN-10.gif)  |         ![image](gifs/safeDQN-10-UniAttack.gif)         |     ![image](gifs/safeDQN-10-STAttack.gif)      | 
| **safeDQN-X** | ![image](gifs/safeDQN-X10.gif) |        ![image](gifs/safeDQN-X10-UniAttack.gif)         |     ![image](gifs/safeDQN-X10-STAttack.gif)     |


## Collaborator

<table>
  <tr>
    <td align="center"><a href="https://github.com/WLeiiiii"><img src="https://github.com/WLeiiiii.png?size=80" width="80px;" alt="Lei Wang"/><br /><sub><b>Lei Wang</b></sub></a><br /><a href="https://github.com/WLeiiii/Gym-ATC-Attack-Project/commits?author=WLeiiiii" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Kaimaoge"><img src="https://github.com/Kaimaoge.png?size=80" width="80px;" alt="Yuankai Wu"/><br /><sub><b>Yuankai Wu</b></sub></a><br /><a href="https://github.com/WLeiiii/Gym-ATC-Attack-Project/commits?author=Kaimaoge" title="Code">💻</a></td>
  </tr>
</table>




