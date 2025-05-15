import os
import sys
import inspect
import logging
import datetime
import json
import asyncio

# Configure logging
log_format = '%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Add a file handler to save logs
def setup_file_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Also add the file handler to the root logger to capture logs from utils.run
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set to INFO to reduce debug messages
    root_logger.addHandler(file_handler)
    
    # Configure all loggers from the utils module to be more concise
    for name in logging.root.manager.loggerDict:
        if name.startswith('utils.'):
            logging.getLogger(name).setLevel(logging.INFO)
    
    logger.info(f"Logging to file: {log_file}")
    return log_file

# Add the parent directory to sys.path to make imports work
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.run import (
    run_experiment_task1,
    save_results_to_csv,
    create_visit_order,
    save_prompts_to_json,
    save_conversation_history
)

# Import configuration
from config import EXPERIMENT_CONFIG

async def process_subject(subject_id, n, subject_seed, strategy, env_order, outdir):
    """Process a single subject asynchronously"""
    logger.info(f"Processing subject {subject_id}/{n}")
    logger.info(f"Using subject_seed={subject_seed}")
    logger.info(f"Strategy for subject {subject_id}: {strategy}")

    # Use a unique conversation history attribute name for isolation
    conversation_history_attr = f'_conversation_history_{subject_seed}'
    
    results, prompts = await run_experiment_task1(
        labels_csv=None,
        seed=subject_seed,
        agent_strategy=strategy,
        external_visit_order=env_order,
        conversation_history_attr=conversation_history_attr
    )
    logger.info(f"Run completed for subject {subject_id}: {len(results)} trials")
    
    # fill subject ID
    for row in results:
        row['subject_id'] = subject_id

    # Save results to CSV
    fname = f"subject_{subject_id}.csv"
    save_results_to_csv(results, outdir=outdir, filename_override=fname)
    
    # Save prompts to JSON
    prompts_fname = f"subject_{subject_id}_prompts.json"
    save_prompts_to_json(prompts, outdir=outdir, filename_override=prompts_fname)
    logger.info(f"Saved results and prompts for subject {subject_id}")
    
    # NOTE: Conversation history is now saved *within* run_experiment_task1
    # The block below is removed as it's redundant and causes warnings.
    # # Save conversation history if using LLM 
    # if strategy == 'llm':
    #     conv_fname = f"subject_{subject_id}_conversation.json"
    #     # This call is redundant because run_experiment_task1 already saves using the specific history_attr
    #     save_conversation_history(outdir=outdir, filename_override=conv_fname, history_attr=conversation_history_attr)
    #     logger.info(f"Saved conversation history for subject {subject_id}")
        
    # Optional: Keep the double-check for safety, although run_experiment_task1 should handle deletion.
    if strategy == 'llm':
        from utils.run import _call_llm_for_choice_async
        if hasattr(_call_llm_for_choice_async, conversation_history_attr):
            # This check might still be useful to catch unexpected errors where deletion failed inside run_experiment_task1
            logger.error(f"CRITICAL ERROR: Conversation history {conversation_history_attr} was NOT properly cleared after run_experiment_task1 for subject {subject_id}!")
    
    return results

async def collect_n_subjects(n=None, seed=42, outdir="logs", use_llm=True, max_concurrent=None):
    """
    Collect data from n subjects for the experiment in parallel using asyncio.
    Each subject completes visits to 3 different casinos, 24 visits per casino.
    Total: 72 trials per subject
    
    Args:
        n: Number of subjects to run
        seed: Base random seed
        outdir: Output directory for results
        use_llm: Whether to use LLM or random strategy
        max_concurrent: Maximum number of subjects to process concurrently
    """
    # Use config values if not specified
    if n is None:
        n = EXPERIMENT_CONFIG["num_subjects"]
    
    if max_concurrent is None:
        max_concurrent = EXPERIMENT_CONFIG["max_concurrent_subjects"]
        
    logger.info(f"Starting parallel collection for {n} subjects with seed={seed}, use_llm={use_llm}")
    logger.info(f"Running with max_concurrent={max_concurrent} subjects at a time")
    
    # build the environment order - shared across all subjects
    env_order = create_visit_order(
        num_casinos=len(EXPERIMENT_CONFIG["casinos"]), 
        visits_per_casino=EXPERIMENT_CONFIG["trials_per_casino"], 
        seed=seed
    )
    logger.info(f"Generated visit order (first 10): {env_order[:10]}...")

    # Create tasks for all subjects but process in batches to limit concurrency
    all_data = []
    strategy = 'llm' if use_llm else 'random'
    
    # Process subjects in batches
    for batch_start in range(1, n+1, max_concurrent):
        batch_end = min(batch_start + max_concurrent - 1, n)
        logger.info(f"Processing batch of subjects {batch_start}-{batch_end}")
        
        # Create tasks for this batch
        tasks = []
        for subject_id in range(batch_start, batch_end + 1):
            subject_seed = seed + subject_id
            task = process_subject(subject_id, n, subject_seed, strategy, env_order, outdir)
            tasks.append(task)
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten and add results to all_data
        for results in batch_results:
            all_data.extend(results)
            
    logger.info(f"Collection complete: {len(all_data)} total trials across {n} subjects")
    return all_data

if __name__=="__main__":
    log_file = setup_file_logging()
    logger.info(f"Starting experiment run. Logs will be saved to {log_file}")
    
    # Check if we're in test mode (fewer subjects and trials)
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if test_mode:
        logger.info("Running in TEST MODE with reduced subjects and trials")
        # Create a smaller environment with just 4 trials per casino
        from utils.run import create_visit_order
        env_order = create_visit_order(
            num_casinos=len(EXPERIMENT_CONFIG["casinos"]), 
            visits_per_casino=4, 
            seed=100
        )
        
        # Run a single subject with this smaller environment
        async def run_test():
            from utils.run import run_experiment_task1
            results, prompts = await run_experiment_task1(
                labels_csv=None,
                seed=100,
                agent_strategy='llm',
                external_visit_order=env_order
            )
            
            # Fill in subject ID
            for row in results:
                row['subject_id'] = 1
                
            # Save results
            save_results_to_csv(results, outdir="logs", filename_override="test_subject.csv")
            save_prompts_to_json(prompts, outdir="logs", filename_override="test_prompts.json")
            logger.info(f"Test completed with {len(results)} trials")
            
        asyncio.run(run_test())
    else:
        # Normal mode - run subjects according to config with asyncio
        asyncio.run(collect_n_subjects(
            seed=100, 
            outdir="logs", 
            use_llm=True,
            max_concurrent=EXPERIMENT_CONFIG["max_concurrent_subjects"]
        ))
        logger.info("Done with all subjects.")
