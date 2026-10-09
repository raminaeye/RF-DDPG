# Continuous Control with DDPG

This project solves continuous-control tasks from Unity ML-Agents with Deep Deterministic Policy Gradient (DDPG), an actor-critic method for continuous action spaces. It was built as Project 2 of the Udacity Deep Reinforcement Learning Nanodegree.

## The task

**Reacher** (the main environment): a double-jointed arm must move its hand to target locations. A reward of +0.1 is given for each step the hand stays in the goal location. The observation space has 33 variables (positions, rotations, velocities, and angular velocities of the arm). Each action is a vector of 4 torques, one per joint motor, clipped to [-1, 1].

The version solved here runs **20 identical agents in parallel**, each in its own copy of the environment. The environment counts as solved when the average score (mean over the 20 agents of their undiscounted episode returns) reaches **+30 averaged over 100 consecutive episodes**.

**Crawler** (bonus environment): a four-armed creature learns to walk forward. Same DDPG setup, tuned separately.

## Method

- **DDPG actor-critic**: the actor maps states to torques; the critic estimates Q-values for state-action pairs and its gradient trains the actor. The critic joins the action vector with the state representation after its first 256-unit layer.
- **Target networks** for actor and critic, soft-updated with tau = 1e-3.
- **Shared replay buffer** (size 100k) across all parallel agents, trained on random minibatches of 256.
- **Ornstein-Uhlenbeck noise** for temporally smooth exploration, gradient-norm clipping on the critic, and batch normalization in both networks.

Key hyperparameters: gamma = 0.99, actor lr = 1e-3, critic lr = 1e-3. See `agent.py` for the full list.

## Repo structure

- `Continuous_Control.ipynb` - train and evaluate the DDPG agent on Reacher (20 agents).
- `Crawler.ipynb` - train and evaluate the DDPG agent on the Crawler environment.
- `agent.py` - the DDPG agent: replay buffer, OU noise, actor/critic updates, target soft updates.
- `model.py` - the actor and critic network architectures.
- `requirements.txt` - Python dependencies.
- `artifact/` - trained weights (`*_best_actor.pth`, `*_best_critic.pth`, plus checkpoints) and demo videos (`after_learn.mov`, `crawler_rf.mov`).
- `reacher/` - the Unity environment binaries used for training (macOS apps).
- `reward_20_agents.png` - training curve for the 20-agent Reacher run.

## How to train

1. Install dependencies: `pip install -r requirements.txt`
2. Get a Unity Reacher environment binary for your platform (the `reacher/` folder ships the macOS builds; Windows and Linux builds are linked in the notebook). Training **requires the Unity ML-Agents Reacher environment**; it cannot run without it.
3. Open `Continuous_Control.ipynb` and run the cells top to bottom. Checkpoints save after every episode, and the best weights are kept once the 100-episode average score passes +30.
4. To watch the trained agent, run the final section of the notebook, which loads `artifact/reacher_best_actor.pth` and renders the arm with graphics on.

The Crawler notebook follows the same flow for the Crawler environment.

## Results

The Reacher agent reaches an average score of +30 over 100 consecutive episodes across the 20 parallel agents:

![Training scores](reward_20_agents.png)

The best trained weights are in `artifact/` (`reacher_best_actor.pth`, `reacher_best_critic.pth`, `crawler_best_actor.pth`, `crawler_best_critic.pth`), and the demo videos show the learned behaviors.

## Notes

- The Unity environment binaries and trained weights are committed to this repo, so a fresh clone is about 514 MB.
- The code targets the legacy Unity ML-Agents Python API (`brain_names`, `vector_observations`); it was written against `unityagents` 0.4.x and is not expected to work with modern ML-Agents releases without changes.
