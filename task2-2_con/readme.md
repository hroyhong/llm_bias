# Asymmetric Learning Experiment with Forced Random First Trials

This repository implements a multi-armed bandit experiment inspired by reinforcement learning paradigms, specifically focusing on learning from chosen outcomes only.

## Experiment Overview

### Core Structure
- **Total Trials:** 96 trials per subject (24 visits to each of 4 casinos).
- **Casinos:** Four types with varying reward probabilities for two machines:
  - Casino 1: [0.25, 0.25] - Low-Low
  - Casino 2: [0.25, 0.75] - Low-High
  - Casino 3: [0.75, 0.25] - High-Low
  - Casino 4: [0.75, 0.75] - High-High
- **Feedback Type:** Agent sees the outcomes of *both* the chosen and the unchosen (counterfactual) machine.
- **Reward Structure:** Configurable rewards (e.g., 1 for win, 0 for loss) defined in `config.py`. The goal is typically to maximize total points.
- **Agent Strategies:** Currently supports 'random' or 'llm' (using a large language model).

### Key Features & Recent Changes
- **Forced Random First Trial:** To mitigate potential initial bias (e.g., LLMs often favoring the first option presented), the very *first* time an agent encounters *each specific casino type* within their 96 trials, a random (50/50) choice between the two machines is forced. This applies to all agents, including the LLM.
- **LLM History Injection:** When the agent strategy is 'llm', the outcome of this forced random first trial is correctly injected into the LLM's conversation history (as a User prompt, Assistant response representing the forced choice, and System outcome message). This ensures the LLM's learning process starts with an unbiased first experience for each casino type.
- **Memory Management:** LLM conversation history size is managed via `MEMORY_CONFIG` in `config.py` (e.g., `max_history_turns`).
- **Strategy Simplification:** Removed baseline strategies `always_first` and `always_second`.

## Experiment Details

### Casino Structure
Each casino has two machines identified by labels (e.g., 'Z' and 'H', loaded from `casino_labels.csv`). The win probabilities are defined in `EXPERIMENT_CONFIG["casinos"]` within `config.py`.

### Trial Sequence
1. The agent (subject) visits casinos according to a randomized `visit_order` (96 trials total, ensuring 24 visits per casino type).
2. **First Visit to Casino Type:** A random 50/50 choice is forced.
3. **Subsequent Visits:** The agent chooses a machine based on the selected `agent_strategy` ('random' or 'llm').
4. The chosen machine delivers a reward based on its probability.
5. The outcomes of *both* the chosen and unchosen (counterfactual) machines are revealed to the agent.
6. **LLM Learning:** If using the 'llm' strategy, the agent uses its conversation history (which includes the forced first trial outcome and the *complete feedback* from subsequent trials) to inform choices.

## Configuration

Key settings are in `config.py`:

```python
# LLM Model Settings
LLM_CONFIG = { ... }

# Memory settings for LLM agent
MEMORY_CONFIG = {
    "max_history_turns": 96, # Max number of User-Assistant-System turns to keep
    "history_overflow_strategy": "truncate_oldest", # How to handle exceeding the limit
    "inform_llm_about_memory": True
}

# Experiment Structure
EXPERIMENT_CONFIG = {
    "total_trials": 96,
    "trials_per_casino": 24,
    "num_subjects": 50, # Number of agents/subjects to run
    "max_concurrent_subjects": 20, # For parallel execution
    "casinos": {
        1: [0.25, 0.25],
        2: [0.25, 0.75],
        3: [0.75, 0.25],
        4: [0.75, 0.75]
    },
    "rewards": {
        "win": 0, # Represents no loss
        "loss": -1 # Represents a loss
    }
}

# System message for LLM agent
SYSTEM_MESSAGE = """... [Your system prompt instructing the LLM] ..."""
```

## Running the Experiment

Use the `experiment/run_multiple.py` script:

```bash
# Run full experiment with LLM agents (using config settings)
python -m experiment.run_multiple

# Run full experiment with Random agents
# (Requires modifying run_multiple.py or adding command-line args to set use_llm=False)
# Example modification in run_multiple.py's __main__ block:
# asyncio.run(collect_n_subjects(use_llm=False, ...))

# Run a quick test mode (fewer trials/subjects, uses LLM)
python -m experiment.run_multiple --test
```

## Output Files

Located in the `logs/` directory (or as configured):
- `subject_[ID].csv`: Trial-by-trial data for each subject, including chosen action, outcome, reward (Win=1, Loss=0), and counterfactual outcome/reward information.
- `subject_[ID]_prompts.json`: Prompts generated for each trial (for LLM or potentially other agents).
- `subject_[ID]_conversation.json`: Full LLM conversation history (if `agent_strategy='llm'`).
- `subject_[ID]_conversation_readable.txt`: Human-readable version of the LLM conversation.
- `experiment_[timestamp].log`: Detailed execution log.

## Notes
- Ensure your `.env` file has the necessary API keys (e.g., `ARK_API_KEY`).
- The forced random first trial ensures a less biased starting point for learning within each casino context.