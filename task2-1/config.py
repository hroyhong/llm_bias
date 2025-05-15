"""
Configuration file for experiment parameters
"""

# LLM settings
LLM_CONFIG = {
    "model": "deepseek-v3-250324",
    "temperature": 0,
    "max_tokens": 150
}

# Memory configuration
MEMORY_CONFIG = {
    # Set to 72 to match exactly the number of trials per subject (24 trials x 3 casinos)
    "max_history_turns": 72,
    
    # How to handle history when it exceeds the limit
    # Options: "truncate_oldest", "summarize"
    "history_overflow_strategy": "truncate_oldest",
    
    # Whether to include the memory management in the system message
    "inform_llm_about_memory": True
}

# Experiment structure
EXPERIMENT_CONFIG = {
    # Number of trials
    "total_trials": 72,
    
    # Number of trials per casino
    "trials_per_casino": 24,
    
    # Number of subjects in normal mode
    "num_subjects": 50,
    
    # Maximum number of subjects to run concurrently
    "max_concurrent_subjects": 20,
    
    # Casino configurations: casino_id -> [prob_option1, prob_option2]
    "casinos": {
        1: [0.5, 0.5],   # Symmetric
        2: [0.75, 0.25], # Asymmetric
        3: [0.25, 0.75]  # Asymmetric reversed
    },
    
    # Reward structure
    "rewards": {
        "win": 1,   # Reward for winning
        "loss": -1  # Reward for losing
    }
}

# System message for LLM
SYSTEM_MESSAGE = """You are going to visit three different casinos (named 1, 2, and 3) 24 times each. Each casino owns two slot machines which stochastically give you either +1 points (a win) or -1 points (a loss) with different win probabilities. Your goal is to maximize the total points won within 72 visits.

Use your memory of past outcomes to make better decisions.

IMPORTANT: For each decision, first provide ONE SENTENCE analyzing your strategy or the casino patterns, then state your choice in the format: 'My choice is: [MACHINE]'.""" 