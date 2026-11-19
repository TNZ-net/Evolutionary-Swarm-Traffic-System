# sumo-swarm

Multi-agent traffic optimisation in SUMO using GRU-PPO agents evolved by a genetic algorithm.

## Setup

```bash
git clone https://github.com/yourname/sumo-swarm
cd sumo-swarm
pip install -r requirements.txt
cp .env.example .env   # fill in your SUMO_HOME path
```

## File map

```
sumo-swarm/
  network/               <- SUMO files (built locally in netedit, uploaded once)
  src/
    config.py            <- all tunable values — change here only
    env.py               <- SUMO multi-agent gym wrapper
    agent.py             <- GRU encoder + PPO policy
    ga.py                <- chromosome, fitness, crossover, mutation, healing
    coordinator.py       <- global KPI aggregator + reward shaper
    utils.py             <- logging, checkpoints, seed
  train.py               <- main training entry point
  notebooks/
    01_baseline.ipynb    <- measure fixed-time baseline (run first)
    02_train.ipynb       <- full GA training on Colab
  checkpoints/           <- auto-created, saved to Drive
```

## Local smoke test (no GPU needed)

```bash
python train.py --episodes 2 --generations 1
```
 
## Resuming after session drop

```bash
python train.py --episodes 5 --generations 50 --resume checkpoints/gen_0025.json
```

## Adjusting hyperparameters

Edit src/config.py only. All modules import from there.
