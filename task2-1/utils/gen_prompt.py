def generate_prompt(casino, visit, outcome_history, machine_opts):
    """
    Generates a prompt that shows the previous outcome first, then the current choice.
    Similar to a human experiment where you're told the result of your last choice,
    then asked to make your next choice.
    """
    prompt = ""
    
    # First, show the LAST outcome if available (not the entire history)
    if outcome_history:
        last_outcome = outcome_history[-1]
        prompt += f"{last_outcome}\n\n"
    
    # Then provide the current situation and choice options
    prompt += f"Casino {casino}, visit {visit}: Choose {machine_opts[0]} or {machine_opts[1]}."
    
    return prompt
