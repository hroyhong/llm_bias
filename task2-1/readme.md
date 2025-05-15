# Factual Learning Experiment

This repository implements a multi-armed bandit experiment based on the Palminteri et al. (2017) "Factual Learning" paradigm, where participants only see the outcome of their chosen option.

## Experiment Overview

### Core Structure
- **Total Trials:** 72 trials per subject (24 visits to each of 3 casinos)
- **Casinos:** 
  - Casino 1: Symmetric condition [0.5, 0.5]
  - Casino 2: Asymmetric condition [0.75, 0.25]
  - Casino 3: Asymmetric reversed [0.25, 0.75]
- **Feedback Type:** Factual learning (only shows outcome of chosen machine)
- **Reward Structure:** Binary outcomes (+1 for win, -1 for loss)

### Important Issue Fixed
- **Memory Management:** The previous implementation had unlimited conversation history (`max_history_turns: -1`), causing excessive memory usage
- **Solution:** Set `max_history_turns: 72` in config.py to limit history to exactly the number of trials needed

## Experiment Details

### Casino Structure
Each casino has two labeled machines (e.g., 'Z' and 'H') with different reward probabilities:
- The symmetric casino has both machines with 50% win rate
- The asymmetric casinos have one machine with 75% win rate and the other with 25%

### Trial Sequence
1. The participant visits casinos in a randomized order (but each casino exactly 24 times)
2. On each visit, they choose one of two machines
3. The chosen machine delivers a reward (+1) or loss (-1)
4. Only the outcome of the chosen machine is revealed (factual learning)
5. This information should be used to make better choices in future visits

### Memory and Learning
- Each subject's conversation history tracks their choices and outcomes
- The LLM should learn which machines have better reward rates in each casino
- Memory is managed appropriately to include only relevant history

## Configuration

The experiment is fully configurable through `config.py`, with these key settings:

```python
# Important memory settings
MEMORY_CONFIG = {
    # Should be exactly 72 (total trials), not unlimited (-1)
    "max_history_turns": 72,
    "history_overflow_strategy": "truncate_oldest"
}

# Casino settings
EXPERIMENT_CONFIG = {
    "total_trials": 72,
    "trials_per_casino": 24,
    "num_subjects": 50,
    "casinos": {
        1: [0.5, 0.5],   # Symmetric 
        2: [0.75, 0.25], # Asymmetric
        3: [0.25, 0.75]  # Asymmetric reversed
    },
    "rewards": {
        "win": 1,
        "loss": -1
    }
}
```

## Running the Experiment

```bash
# Run full experiment (50 subjects)
python -m experiment.run_multiple

# Run a test with reduced trials
python -m experiment.run_multiple --test
```

## Output Files

Each run produces:
- CSV files with trial-by-trial data
- JSON files with the prompts used in each trial
- JSON files with the complete conversation history
- Detailed logs of the experiment

The conversation history properly maintains the context needed for the LLM to learn, without exceeding the required number of trials.