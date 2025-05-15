import os
import csv
import json
import random
import datetime
import logging
import re
import sys
import asyncio
from dotenv import load_dotenv
from utils.gen_prompt import generate_prompt

# Add parent directory to path for importing config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import configuration
from config import EXPERIMENT_CONFIG, LLM_CONFIG, SYSTEM_MESSAGE, MEMORY_CONFIG

# For the LLM
from openai import OpenAI
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
def load_env():
    # Get the project root directory (2 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    env_path = os.path.join(project_root, '.env')
    
    # Load .env file if it exists
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")
        # Try looking in the parent directory as a fallback
        parent_env_path = os.path.join(os.path.dirname(project_root), '.env')
        if os.path.exists(parent_env_path):
            load_dotenv(parent_env_path)
            logger.info(f"Loaded environment variables from parent directory: {parent_env_path}")
        else:
            logger.error(f"No .env file found in either {env_path} or {parent_env_path}")

def _ark_client():
    # Load environment variables
    load_env()
    
    # Get API key
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY not found in environment variables. Please check your .env file.")
    
    logger.info("Successfully initialized ARK client with API key")
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key
    )

def _ark_async_client():
    # Load environment variables
    load_env()
    
    # Get API key
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY not found in environment variables. Please check your .env file.")
    
    logger.info("Successfully initialized async ARK client with API key")
    return AsyncOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key
    )

def _default_casino_labels_path():
    base_dir = os.path.dirname(__file__)
    task1_dir = os.path.join(base_dir, '..')
    return os.path.abspath(os.path.join(task1_dir, 'casino_labels.csv'))

def load_casino_labels(csv_file=None):
    if csv_file is None:
        csv_file = _default_casino_labels_path()

    labels = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            c = int(row[0])
            lab1 = row[1]
            lab2 = row[2]
            labels[c] = [lab1, lab2]
    return labels

# Casino probabilities now loaded from config
ENV_PROBS = EXPERIMENT_CONFIG["casinos"]

def create_visit_order(num_casinos=4, visits_per_casino=24, seed=None):
    """Create a randomized order of casino visits"""
    if seed is not None:
        random.seed(seed)
    visits = []
    for c in range(1, num_casinos+1):
        visits.extend([c]*visits_per_casino)
    random.shuffle(visits)
    return visits

def save_conversation_summary(history, outdir='logs', filename_override=None):
    """
    Creates a human-readable summary of the conversation history with clear
    indication of prompts, responses, and outcomes.
    """
    if not history:
        logger.warning("No conversation history to summarize")
        return
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if filename_override:
        fname = filename_override.replace('.json', '_readable.txt')
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"conversation_summary_{stamp}.txt"
        
    path = os.path.join(outdir, fname)
    
    # Create a readable summary version
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== CONVERSATION HISTORY SUMMARY ===\n\n")
        f.write(f"Total messages: {len(history)}\n\n")
        
        # Print system message first
        for i, msg in enumerate(history):
            if msg["role"] == "system" and i == 0:
                f.write("=== SYSTEM INSTRUCTIONS ===\n")
                f.write(f"{msg['content']}\n\n")
                break
        
        # Process the conversation in order
        trial_num = 1
        for i in range(1, len(history)):
            msg = history[i]
            
            # Format based on message role
            if msg["role"] == "user":
                f.write(f"=== TRIAL {trial_num} ===\n")
                f.write("PROMPT:\n")
                f.write(f"{msg['content']}\n\n")
                trial_num += 1
            elif msg["role"] == "assistant":
                f.write("RESPONSE:\n")
                f.write(f"{msg['content']}\n\n")
            elif msg["role"] == "system" and i > 0:  # Skip first system message
                f.write("OUTCOME:\n")
                f.write(f"{msg['content']}\n\n")
                f.write("-" * 50 + "\n\n")
    
    logger.info(f"Created readable conversation summary at {path}")
    return path

def save_conversation_history(outdir='logs', filename_override=None, history_attr='_conversation_history'):
    """
    Save the full conversation history to a JSON file.
    This allows for analysis of the LLM's decision-making process.
    """
    if not hasattr(_call_llm_for_choice_async, history_attr):
        logger.warning("No conversation history found to save")
        return
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if filename_override:
        fname = filename_override
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"conversation_history_{stamp}.json"
        
    path = os.path.join(outdir, fname)
    
    # Check number of user messages to ensure we don't exceed max trials
    user_messages = sum(1 for msg in getattr(_call_llm_for_choice_async, history_attr) if msg["role"] == "user")
    if user_messages > EXPERIMENT_CONFIG["total_trials"]:
        logger.error(f"ERROR: Found {user_messages} user messages in history, but max should be {EXPERIMENT_CONFIG['total_trials']}")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(getattr(_call_llm_for_choice_async, history_attr), f, indent=2)
    
    logger.info(f"Saved conversation history ({len(getattr(_call_llm_for_choice_async, history_attr))} messages) to {path}")
    
    # Also create a human-readable summary version
    summary_path = save_conversation_summary(getattr(_call_llm_for_choice_async, history_attr), outdir, filename_override)
    logger.info(f"Created human-readable summary at {summary_path}")
    
    # Explicitly delete the conversation history to start fresh for next subject
    delattr(_call_llm_for_choice_async, history_attr)
    logger.info("Deleted conversation history after saving")
    
    return path

# Create semaphore to control API access
_API_SEMAPHORE = asyncio.Semaphore(EXPERIMENT_CONFIG.get("max_concurrent_api_calls", 50)) # Use config value or default

async def run_experiment_task1(labels_csv=None,
                         seed=None,
                         agent_strategy='random',
                         external_visit_order=None,
                         conversation_history_attr='_conversation_history'):
    """
    Run an experiment with 96 trials across 4 casino types:
    - Casino 1: Low-Low [0.25, 0.25]
    - Casino 2: Low-High [0.25, 0.75]
    - Casino 3: High-Low [0.75, 0.25]
    - Casino 4: High-High [0.75, 0.75]
    
    Each casino has 24 trials for a total of 96 trials per subject
    # Points are 0 for no loss and -1 for a loss. Goal is to minimize total loss.
    
    Args:
        labels_csv: Path to CSV file with casino labels
        seed: Random seed for reproducibility (used as subject ID)
        agent_strategy: Strategy to use ('random', 'llm', 'always_first', 'always_second')
        external_visit_order: Predefined visit order to use (if None, will create one)
        conversation_history_attr: Name of attribute to store conversation history
                                  (used to isolate memory between parallel runs)
    """
    # Import random explicitly (needed for multiprocessing to avoid reference errors)
    import random as random_module
    
    # Use a unique conversation history attribute name for this run
    # This ensures memory isolation when multiple subjects run in parallel
    if conversation_history_attr == '_conversation_history':
        # If no custom attribute name is provided, use default with seed
        if seed is not None:
            conversation_history_attr = f'_conversation_history_{seed}'
        else:
            # Random suffix as fallback
            suffix = random_module.randint(10000, 99999)
            conversation_history_attr = f'_conversation_history_{suffix}'
    
    logger.info(f"======= STARTING EXPERIMENT (Seed: {seed}, Strategy: {agent_strategy}) =======")
    logger.info(f"Using History attribute: {conversation_history_attr}")
    if seed is not None:
        random_module.seed(seed)

    casino_labels = load_casino_labels(labels_csv)
    logger.info(f"Loaded casino labels: {casino_labels}")
    
    # Create or use visit order
    if external_visit_order is not None:
        visit_order = external_visit_order
    else:
        visit_order = create_visit_order(
            num_casinos=len(ENV_PROBS), 
            visits_per_casino=EXPERIMENT_CONFIG["trials_per_casino"],
            seed=seed
        )
    logger.info(f"Visit order created: {visit_order[:5]}... ({len(visit_order)} total visits)")

    visit_counter = {c: 0 for c in ENV_PROBS.keys()}
    outcome_history = []
    results = []
    prompts_data = []
    first_visit_done = {c: False for c in ENV_PROBS.keys()} # Track first visit per casino

    # Build an LLM client once if needed
    llm_client = None # Keep for potential future sync use?
    async_llm_client = None
    if agent_strategy=='llm':
        logger.info("Initializing LLM client...")
        # llm_client = _ark_client() # Currently unused
        async_llm_client = _ark_async_client()
        # Reset conversation history for this subject/seed AT THE START
        if hasattr(_call_llm_for_choice_async, conversation_history_attr):
            delattr(_call_llm_for_choice_async, conversation_history_attr)
            logger.info(f"Cleared previous conversation history attribute: {conversation_history_attr}")
        # Initialize fresh history here regardless of previous state
        setattr(_call_llm_for_choice_async, conversation_history_attr, [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ])
        logger.info(f"Initialized new conversation history for {conversation_history_attr}")

    # Use enumerate starting from 1 for trial_idx
    for trial_idx, c in enumerate(visit_order, 1):
        visit_counter[c]+=1
        curr_visit = visit_counter[c]
        machine_opts = casino_labels[c]  # e.g. ["H","M"]
        
        logger.info(f"\n===== TRIAL {trial_idx} =====")
        logger.info(f"Casino {c}, Visit {curr_visit}, Options: {machine_opts}")

        # Check if this is the very first visit FOR THIS CASINO TYPE in this run
        is_first_visit_for_casino = not first_visit_done[c]
        llm_chose_this_trial = False # Flag to track if LLM was actually called

        # build the text prompt - depends on outcome_history from PREVIOUS trials
        prompt_str = generate_prompt(c, curr_visit, outcome_history, machine_opts)
        logger.debug(f"Generated prompt for Trial {trial_idx}: {prompt_str}") # Changed to debug

        # --- Pick the machine ---
        if is_first_visit_for_casino:
            # Force random choice for the first visit to this specific casino
            chosen_idx = random_module.choice([0, 1])
            logger.info(f"FIRST VISIT to Casino {c}: Forced random choice. Index: {chosen_idx}")
            first_visit_done[c] = True # Mark as visited for this run
        elif agent_strategy=='random':
            chosen_idx = random_module.choice([0,1])
            logger.info(f"Random strategy chose index {chosen_idx}")
        elif agent_strategy=='llm':
            # LLM makes the choice (only if not the first visit for this casino)
            logger.info(f"Calling LLM for Casino {c} (Visit {curr_visit})...")
            
            # The async function handles adding the user prompt and assistant response to its history_attr
            chosen_label_from_llm = await _call_llm_for_choice_async(
                client=async_llm_client, 
                prompt_text=prompt_str, # Pass the generated prompt based on past outcomes
                machine_opts=machine_opts, 
                all_outcomes=outcome_history, # Pass for context if needed by called function? No, it uses history_attr
                history_attr=conversation_history_attr
            )
            chosen_idx = 0 if chosen_label_from_llm==machine_opts[0] else 1
            logger.info(f"LLM chose: {chosen_label_from_llm} (index {chosen_idx})")
            llm_chose_this_trial = True
        else: # Default fallback strategy
            logger.warning(f"Unknown agent_strategy '{agent_strategy}', defaulting to random.")
            chosen_idx = random_module.choice([0,1])
            logger.info(f"Default (random) strategy chose index {chosen_idx}")

        # --- Determine Outcome ---
        chosen_label = machine_opts[chosen_idx]
        prob_of_no_loss = ENV_PROBS[c][chosen_idx] # Probability of getting 0 points
        # Use reward values from config
        reward = EXPERIMENT_CONFIG["rewards"]["win"] if random_module.random() < prob_of_no_loss else EXPERIMENT_CONFIG["rewards"]["loss"]
        chosen_outcome_numeric = 1 if reward == EXPERIMENT_CONFIG["rewards"]["win"] else 0 # 1 for win, 0 for loss

        # --- Determine Counterfactual Outcome --- START
        unchosen_idx = 1 - chosen_idx
        unchosen_label = machine_opts[unchosen_idx]
        prob_of_no_loss_unchosen = ENV_PROBS[c][unchosen_idx]
        counterfactual_reward = EXPERIMENT_CONFIG["rewards"]["win"] if random_module.random() < prob_of_no_loss_unchosen else EXPERIMENT_CONFIG["rewards"]["loss"]
        counterfactual_outcome_numeric = 1 if counterfactual_reward == EXPERIMENT_CONFIG["rewards"]["win"] else 0 # 1 for win, 0 for loss
        # --- Determine Counterfactual Outcome --- END

        logger.info(f"OUTCOME: Chosen: {chosen_label} -> Reward {reward}. Counterfactual: {unchosen_label} -> Reward {counterfactual_reward}")

        # Create the outcome text that will be shown via prompt history / added to conversation history
        # --- Modify outcome message for LLM --- START
        chosen_outcome_term = "win" if chosen_outcome_numeric == 1 else "loss"
        counterfactual_outcome_term = "win" if counterfactual_outcome_numeric == 1 else "loss"
        outcome_line = (
            f"Outcome: You chose Machine {chosen_label} and experienced a {chosen_outcome_term} "
            f"(Reward: {reward}). "
            f"The other machine, Machine {unchosen_label}, would have resulted in a {counterfactual_outcome_term} "
            f"(Reward: {counterfactual_reward})."
        )
        # --- Modify outcome message for LLM --- END

        outcome_history.append(outcome_line) # Append to simple list for prompt generation
        
        # --- Update LLM Conversation History ---
        # This needs to happen *after* the choice and outcome are determined for the current trial
        if agent_strategy == 'llm':
            # Ensure the history object still exists (it should, initialized above)
            if hasattr(_call_llm_for_choice_async, conversation_history_attr):
                history_list = getattr(_call_llm_for_choice_async, conversation_history_attr)
                
                if is_first_visit_for_casino:
                    # Manually add the User prompt, fake Assistant response (forced choice), and System outcome
                    # This injects the forced trial into the LLM's memory correctly
                    
                    # Prepare the prompt that _call_llm_for_choice_async expects format-wise
                    modified_llm_prompt = f"{prompt_str}\n\nPlease provide ONE SENTENCE analyzing your strategy, then state your choice in the format: 'My choice is: [MACHINE]'."
                    
                    history_list.append({"role": "user", "content": modified_llm_prompt})
                    # Represent the forced choice as if the LLM stated it
                    history_list.append({"role": "assistant", "content": f"My choice is: {chosen_label}"}) 
                    history_list.append({"role": "system", "content": f"Outcome: {outcome_line}"})
                    logger.info(f"Added FORCED trial U/A/S to LLM history for Casino {c}")
                
                elif llm_chose_this_trial:
                    # LLM was called, _call_llm_for_choice_async added User/Assistant.
                    # We only need to add the System outcome message here.
                    history_list.append({"role": "system", "content": f"Outcome: {outcome_line}"})
                    logger.info(f"Added LLM trial outcome (System msg) to LLM history for Casino {c}")
                
                # No action needed if agent_strategy is 'llm' but LLM wasn't called (e.g., if logic changes)
                
            else:
                 logger.error(f"LLM history attribute {conversation_history_attr} lost during run! Cannot add outcome.")


        # --- Record Results ---
        results.append({
            "subject_id": seed, # Store seed as subject_id for now
            "trial_index": trial_idx,
            "casino": c,
            "visit_in_casino": curr_visit,
            "chosen_machine_label": chosen_label,
            "reward": reward,
            "prob_of_no_loss": prob_of_no_loss, # Renamed from win_probability
            "outcome_numeric": chosen_outcome_numeric,
            "unchosen_machine_label": unchosen_label,
            "prob_of_no_loss_unchosen": prob_of_no_loss_unchosen,
            "counterfactual_reward": counterfactual_reward,
            "counterfactual_outcome_numeric": counterfactual_outcome_numeric,
            "prompt_used": prompt_str, # The prompt generated based on outcome_history
            "llm_history_context_turns": len(getattr(_call_llm_for_choice_async, conversation_history_attr, [])) // 3 if agent_strategy == 'llm' else 0 # Rough estimate
        })

        prompts_data.append({
            "trial_index": trial_idx,
            "prompt": prompt_str
        })

    logger.info(f"\n======= EXPERIMENT COMPLETED (Seed: {seed}) =======")
    logger.info(f"Total trials run: {len(results)}")
    
    # Save conversation history if we used LLM
    if agent_strategy == 'llm':
        # Construct filename including seed
        history_filename = f"conversation_history_seed{seed}.json" if seed is not None else f"conversation_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        history_file = save_conversation_history(
            outdir='logs', 
            filename_override=history_filename,
            history_attr=conversation_history_attr
        )
        logger.info(f"Saved conversation history to {history_file}")
    
    return results, prompts_data

async def _call_llm_for_choice_async(client, prompt_text, machine_opts, all_outcomes=None, history_attr='_conversation_history'):
    """
    Async version of _call_llm_for_choice that uses semaphores to limit concurrent API calls.
    Uses the conversation history stored in `history_attr`.
    Adds the user prompt and its own response to the history.
    Expects the calling function to add the subsequent system/outcome message.
    """
    try:
        # Create a modified prompt with request for analysis and choice format
        options_str = " or ".join(machine_opts)
        # Use the passed prompt_text which already includes context from previous trials
        modified_prompt = f"{prompt_text}\n\nPlease provide ONE SENTENCE analyzing your strategy, then state your choice in the format: 'My choice is: [MACHINE]'."
        
        # Log only essential info for the trial
        logger.debug(f"LLM getting prompt (modified): {modified_prompt}") # Changed to debug
        
        # Ensure conversation history attribute exists (should be initialized by caller)
        if not hasattr(_call_llm_for_choice_async, history_attr):
            # This is an error state - history should exist
             logger.error(f"LLM history attribute {history_attr} not found when expected!")
             # Fallback: Initialize here, but context is lost
             setattr(_call_llm_for_choice_async, history_attr, [
                 {"role": "system", "content": SYSTEM_MESSAGE}
             ])
             logger.warning("Re-initialized lost history, context may be incorrect.")
        
        current_history = getattr(_call_llm_for_choice_async, history_attr)

        # Add the current prompt to the conversation history *before* API call
        current_history.append({"role": "user", "content": modified_prompt})
        logger.debug(f"Added user prompt to history {history_attr}")

        # --- Memory Management ---
        max_history_turns = MEMORY_CONFIG.get("max_history_turns", 0) # Default 0 means unlimited
        if max_history_turns > 0:
            # Count pairs of user/assistant/system messages roughly representing turns
            # History structure: [Sys, U, A, S, U, A, S, ...] or [Sys, U, A(forced), S, U, A, S, ...]
            # Count User messages as a proxy for turns completed AFTER system message
            user_messages_count = sum(1 for msg in current_history[1:] if msg["role"] == "user")
            
            # If we exceed the limit (using user messages as proxy for turns)
            if user_messages_count > max_history_turns:
                overflow_strategy = MEMORY_CONFIG.get("history_overflow_strategy", "truncate_oldest")
                logger.info(f"History ({user_messages_count} user turns) exceeds limit ({max_history_turns}), applying '{overflow_strategy}' strategy")
                
                if overflow_strategy == "truncate_oldest":
                    system_msg = current_history[0] # Keep the first system message
                    history_after_system = current_history[1:]
                    
                    # Calculate how many full turns (U/A/S triples, or U/A pairs if last S is missing) to remove
                    turns_to_remove = user_messages_count - max_history_turns
                    # Each turn consists of U, A, S messages (3 items), estimate messages to remove
                    # Be careful with off-by-one if the last turn doesn't have S yet.
                    # Let's remove messages in groups corresponding to turns. Find the index of the (turns_to_remove)-th 'user' message.
                    user_indices = [i for i, msg in enumerate(history_after_system) if msg["role"] == "user"]
                    
                    if turns_to_remove < len(user_indices):
                        # Index in history_after_system corresponding to the start of the turn AFTER the ones we remove
                        start_index_to_keep = user_indices[turns_to_remove] 
                        new_history_after_system = history_after_system[start_index_to_keep:]
                        
                        # Reconstruct history
                        new_full_history = [system_msg] + new_history_after_system
                        setattr(_call_llm_for_choice_async, history_attr, new_full_history)
                        logger.info(f"Truncated oldest {turns_to_remove} turn(s) from history {history_attr}")
                        # Update reference for API call
                        current_history = new_full_history 
                    else:
                         logger.warning(f"Calculation error during truncation, could not remove {turns_to_remove} turns.")

                elif overflow_strategy == "summarize":
                    # Summarization not implemented - fallback to truncation
                    logger.warning("Summarize strategy not implemented, falling back to truncate_oldest")
                    # (Duplicate truncation logic - consider refactoring later)
                    system_msg = current_history[0] 
                    history_after_system = current_history[1:]
                    turns_to_remove = user_messages_count - max_history_turns
                    user_indices = [i for i, msg in enumerate(history_after_system) if msg["role"] == "user"]
                    if turns_to_remove < len(user_indices):
                        start_index_to_keep = user_indices[turns_to_remove] 
                        new_history_after_system = history_after_system[start_index_to_keep:]
                        new_full_history = [system_msg] + new_history_after_system
                        setattr(_call_llm_for_choice_async, history_attr, new_full_history)
                        logger.info(f"Truncated oldest {turns_to_remove} turn(s) from history {history_attr} (fallback from summarize)")
                        current_history = new_full_history
                    else:
                         logger.warning(f"Calculation error during fallback truncation.")
        
        # Log the current conversation structure (types only) for debugging before call
        history_types = [f"{i}: {msg['role']}" for i, msg in enumerate(current_history)]
        logger.debug(f"Conversation structure ({len(current_history)} msgs) for API call: {history_types}")
        
        # --- API Call ---
        ans = ""
        async with _API_SEMAPHORE:
            active_tasks = len(asyncio.all_tasks()) # Approximation of concurrency
            # Simplified logging to avoid internal attribute errors
            logger.info(f"Acquired API semaphore for {history_attr} (Max: {EXPERIMENT_CONFIG.get('max_concurrent_api_calls', 50)}, Active Tasks: ~{active_tasks})") 
            start_time = asyncio.get_event_loop().time()
            try:
                completion = await client.chat.completions.create(
                    model=LLM_CONFIG["model"],
                    messages=current_history, # Use potentially truncated history
                    temperature=LLM_CONFIG["temperature"],
                    max_tokens=LLM_CONFIG["max_tokens"]
                )
                ans = completion.choices[0].message.content
            finally:
                 end_time = asyncio.get_event_loop().time()
                 logger.info(f"Released API semaphore for {history_attr}. Call duration: {end_time - start_time:.2f}s")

        # --- Process Response ---
        # Log type and value before processing
        logger.debug(f"Raw API answer type: {type(ans)}, value: {str(ans)[:100]}...") 
        response_content = ans.strip() if isinstance(ans, str) else "" # Ensure ans is string before strip
        
        if not response_content: # Handle case where API returns empty
             logger.error(f"LLM ({history_attr}) returned empty response. Raw answer: '{ans}'")
             # Fallback strategy needed here
             return machine_opts[0]

        # Log the raw response *before* adding to history and parsing
        logger.info(f"LLM raw response ({history_attr}): {response_content}")
        # Add the assistant's response *before* attempting to parse it
        current_history.append({"role": "assistant", "content": response_content})
        logger.debug(f"Added assistant response to history {history_attr}")

        # --- Parse Choice ---
        # Updated parsing logic to handle analysis + choice format
        machine_found = False
        parsed_choice = None

        # 1. Check for exact format "My choice is: [MACHINE]" or "My choice is: MACHINE" (case-insensitive)
        # Search the ENTIRE response_content. Prioritize this format.
        match = re.search(r"My choice is:\s*\[?(\w+)\]?", response_content, re.IGNORECASE)
        if match:
            potential_choice = match.group(1)
            if potential_choice.upper() in [opt.upper() for opt in machine_opts]: # Case-insensitive check against options
                # Find the matching option while preserving its original case
                parsed_choice = next((opt for opt in machine_opts if opt.upper() == potential_choice.upper()), None)
                parsed_choice = potential_choice
                machine_found = True
                logger.info(f"Parsed choice using 'My choice is: [MACHINE]' format ({history_attr}): {parsed_choice}")

        # 2. If not found, check for simpler patterns like "choice is M", "choose M", etc., prioritizing later occurrences
        if not machine_found:
            best_match_pos = -1
            for opt in machine_opts:
                patterns = [
                    rf"\bchoice is:\s*{re.escape(opt)}\b",
                    rf"\bchoice is\s+{re.escape(opt)}\b",
                    rf"\bchoose\s+{re.escape(opt)}\b",
                    rf"\bchoosing\s+{re.escape(opt)}\b",
                    rf"\bpick\s+{re.escape(opt)}\b",
                    rf"\bselect\s+{re.escape(opt)}\b",
                    rf":\s*{re.escape(opt)}\b",
                    # Look for the option potentially at the end of the string or before punctuation
                    rf"\b{re.escape(opt)}[.!?]?$", 
                ]
                for pattern in patterns:
                    # Find the last occurrence
                    for m in re.finditer(pattern, response_content, re.IGNORECASE):
                         if m.start() > best_match_pos:
                              best_match_pos = m.start()
                              parsed_choice = opt
                              machine_found = True
            if machine_found:
                 logger.info(f"Parsed choice using keyword pattern: {parsed_choice}")


        # 3. If still not found, look for the last occurrence of any valid machine option label in the response
        if not machine_found:
             best_match_pos = -1
             for opt in machine_opts:
                 # Find last occurrence of the option as a whole word, case-insensitive
                 for m in re.finditer(rf"\b{re.escape(opt)}\b", response_content, re.IGNORECASE):
                      if m.start() > best_match_pos:
                           best_match_pos = m.start()
                           parsed_choice = opt
                           machine_found = True
             if machine_found:
                  logger.info(f"Parsed choice by finding last valid option mention: {parsed_choice}")


        # 4. Default / Fallback
        if not machine_found:
            parsed_choice = machine_opts[0] # Default to first option
            logger.warning(f"LLM response '{response_content}' didn't yield clear choice. Defaulting to: {parsed_choice}")
        
        return parsed_choice # Return the parsed choice label
        
    except Exception as e: # Catch API errors or processing/parsing errors
        # Log the exception details
        logger.exception(f"Error during LLM call or processing for {history_attr}: {e}")

        # NOTE: If the error happened AFTER adding the assistant message, the raw response is already in history.

        logger.warning(f"Returning default first option '{machine_opts[0]}' due to error.")

        # Return the default choice
        return machine_opts[0] 

# Define _call_llm_for_choice as an alias to _call_llm_for_choice_async for compatibility (if needed elsewhere)
_call_llm_for_choice = _call_llm_for_choice_async

def save_results_to_csv(results, outdir='logs', filename_override=None):
    if not results:
         logger.warning("No results data to save.")
         return None
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if filename_override:
        fname = filename_override
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"2afc_task1_results_{stamp}.csv"
    path = os.path.join(outdir, fname)

    # Define fieldnames including the new ones
    fieldnames = ["subject_id","trial_index","casino","visit_in_casino",
                  "chosen_machine_label","reward","prob_of_no_loss", 
                  "outcome_numeric", "unchosen_machine_label", "prob_of_no_loss_unchosen",
                  "counterfactual_reward", "counterfactual_outcome_numeric",
                  "prompt_used", "llm_history_context_turns"]
    # Preprocess results to ensure all keys exist
    processed_results = []
    for row in results:
        processed_row = {key: row.get(key, None) for key in fieldnames}
        processed_results.append(processed_row)

    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader()
        for row in processed_results:
            w.writerow(row)
    print(f"Saved trial data to {path}")

def save_prompts_to_json(prompts_data, outdir='logs', filename_override=None):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if filename_override:
        fname = filename_override
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"2afc_task1_prompts_{stamp}.json"
        
    path = os.path.join(outdir, fname)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(prompts_data,f,indent=2)
    logger.info(f"Saved prompts to {path}")
