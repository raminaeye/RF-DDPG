# RF-DDPG

A DDPG actor-critic agent in PyTorch that learns continuous control in Unity ML-Agents. It solves the 20-arm Reacher task and is also run on Crawler. Built for the Udacity Deep RL Nanodegree.

## What's inside

- `Continuous_Control.ipynb` trains and runs the Reacher agent
- `agent.py` and `model.py` hold the agent and networks
- `artifact/` holds trained weights and videos

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Continuous_Control.ipynb
```

Use Python 3.6, which the `unityagents` package needs.

## Related tutorial

- [Deep Deterministic Policy Gradient](https://raminaeye.github.io/ddpg.html)
