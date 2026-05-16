# RLlib SAC Discrete Proof

This folder contains a small proof that the Ray RLlib SAC implementation in
this workspace supports `Discrete` action spaces by default.

The check is intentionally minimal:
- the environment is a one-step bandit
- the observation is always zero
- one discrete action is optimal
- SAC should learn to prefer that action quickly

Run it from the repo root:

```bash
python proofs/rllib_sac_discrete/sac_discrete_proof.py --iterations 10
```

Useful knobs:
- `--num-actions` controls the size of the discrete action space
- `--optimal-action` changes which action gets the reward
- `--iterations` controls how long the proof run lasts

