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
    # Set to 96 to match exactly the number of trials per subject (24 trials x 4 casinos)
    "max_history_turns": 96,
    
    # How to handle history when it exceeds the limit
    # Options: "truncate_oldest", "summarize"
    "history_overflow_strategy": "truncate_oldest",
    
    # Whether to include the memory management in the system message
    "inform_llm_about_memory": True
}

# Experiment structure
EXPERIMENT_CONFIG = {
    # Number of trials
    "total_trials": 96,
    
    # Number of trials per casino
    "trials_per_casino": 24,
    
    # Number of subjects in normal mode
    "num_subjects": 50,
    
    # Maximum number of subjects to run concurrently
    "max_concurrent_subjects": 20,
    
    # Casino configurations: casino_id -> [prob_option1, prob_option2]
    "casinos": {
        1: [0.25, 0.25],   # Low-Low
        2: [0.25, 0.75],   # Low-High
        3: [0.75, 0.25],   # High-Low
        4: [0.75, 0.75]    # High-High
    },
    
    # Reward structure
    "rewards": {
        "win": 1,   # Reward for winning
        "loss": 0  # Reward for losing (no loss)
    }
}

# System message for LLM
SYSTEM_MESSAGE = """You are going to visit four different casinos (named 1, 2, 3, and 4) 24 times each. Each casino owns two slot machines which stochastically give you either 1 point (a win) or 0 points (a loss) with different win probabilities. Your goal is to maximize the total points gained within 96 visits.

After each choice, you will see the outcomes of both the machine you chose AND the machine you didn't choose. Use your memory of past outcomes from both chosen and unchosen machines to make better decisions.

IMPORTANT: For each decision, first provide ONE SENTENCE analyzing your strategy or the casino patterns, then state your choice in the format: 'My choice is: [MACHINE]'.""" 