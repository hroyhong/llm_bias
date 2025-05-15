# LLM Mirrors Human Biases but Diverge in Contextual Valuation During Reinforcement Learning - Supplementary Materials

This repository contains the supplementary materials, including code and potentially data, for the research paper: **"LLM Mirrors Human Biases but Diverge in Contextual Valuation During Reinforcement Learning"**.

## Abstract of the Paper

Large Language Models (LLMs) increasingly engage in decision-making, yet whether their learning processes mirror human cognitive biases remains unclear. Here, we compare DeepSeek V3 LLM with human participants across two instrumental reinforcement learning studies. We find that while LLMs, like humans, exhibit robust confirmation bias—preferentially integrating evidence that supports current choices—they markedly diverge in context-dependent value encoding. Unlike humans, whose valuation of neutral outcomes is strongly modulated by the surrounding reward or punishment frame, LLMs display a more 'absolute' valuation. Computational modeling of learning rates substantiates these distinct 'learning styles'. This juxtaposition of shared biases in evidence integration but differing mechanisms for contextual value representation offers critical insights into the emergent decision-making profiles of advanced AI, highlighting fundamental differences from human cognition.

## About This Repository

This repository provides the code used for the experiments detailed in the paper. The study compares the learning behavior of DeepSeek V3 LLM with human participants in instrumental reinforcement learning tasks.

The main findings highlight:
*   Both LLMs and humans show confirmation bias.
*   LLMs differ from humans in context-dependent value encoding, showing a more 'absolute' valuation.

## Code Structure

The code and potentially data for the experiments described in the paper are organized into the following main directories, based on your updated folder names and the individual README files within them:

*   `task1-1/`: Corresponds to **Study 1, Partial Feedback condition**. This involves symmetric +1/-1 point outcomes, where the agent only sees the outcome of their chosen option (factual learning). (Based on `task1-1/readme.md`)
*   `task1-2/`: Corresponds to **Study 1, Complete Feedback condition**. This involves symmetric +1/-1 point outcomes, where the agent sees outcomes of both chosen and unchosen options. (Based on `task1-2/readme.md`)
*   `task2-1_con/`: Corresponds to **Study 2, Punishment Frame (0/-1) with Partial Feedback**. The agent learns in a context with 0 (no loss) or -1 (loss) outcomes and only sees the outcome of their chosen option. The `_con` suffix likely denotes "context" from Study 2. (Based on `task2-1_con/readme.md`)
*   `task2-2_con/`: Corresponds to **Study 2, Reward Frame (+1/0) with Complete Feedback**. The agent learns in a context with +1 (win) or 0 (no win) outcomes and sees outcomes for both chosen and unchosen options. The `_con` suffix likely denotes "context" from Study 2. (Based on `task2-2_con/readme.md` which describes complete feedback, assuming the reward structure matches the +1/0 frame for Study 2 despite a slightly contradictory example in its config snippet).
*   `task1_analysis/`: Contains analysis scripts, and resulting data related to **Study 1**.
*   `task2_analysis/`: Contains analysis scripts, and resulting data related to **Study 2**.

The paper details two main studies:
*   **Study 1**: Focused on learning dynamics and confirmation bias in symmetric +1/-1 outcome contexts, with Partial and Complete feedback conditions.
*   **Study 2**: Focused on context-dependent value encoding using a Reward Frame (+1/0 outcomes) and a Punishment Frame (0/-1 outcomes), also with Partial and Complete feedback conditions.

The organization above covers these specific conditions from the studies based on the current folder structure. Please refer to the "Methods" section of the paper and the individual README files in each subdirectory for a comprehensive description of the experimental design, parameters, and specific configurations.

## How to Use

Each experimental condition directory (e.g., `task2-1/`, `task2-2/`, `task1-1_con/`, `task1-2_con/`) appears to be a self-contained experiment setup.

**General Instructions (based on common patterns in sub-directory READMEs):**

1.  **Navigate to the specific experiment directory:**
    ```bash
    cd path/to/specific_task_directory 
    # e.g., cd task2-1/
    ```
2.  **Set up Environment (if applicable):**
    *   Ensure you have Python installed.
    *   Create and activate a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install dependencies: Each directory contains a `requirements.txt` file.
        ```bash
        pip install -r requirements.txt
        ```
    *   **API Keys**: For experiments involving LLMs, you will likely need to set up API keys. The `task1-1_con/readme.md` mentions ensuring your `.env` file has the necessary API keys (e.g., `ARK_API_KEY`). Create a `.env` file in the respective experiment directory if it doesn't exist and add your keys there.
        Example `.env` content:
        ```
        ARK_API_KEY=your_api_key_here
        # Add other necessary keys
        ```
        *(Remember to add `.env` to your `.gitignore` file if it's not already there to avoid committing secrets.)*

3.  **Configuration:**
    *   Experiment parameters (e.g., number of trials, casino setups, LLM model settings, memory configuration) are typically defined in a `config.py` file within each experiment directory. Review and modify this file as needed.

4.  **Running the Experiment:**
    *   The experiments are generally run using a script, often `experiment/run_multiple.py`.
    *   To run the full experiment (e.g., for 50 subjects as configured):
        ```bash
        python -m experiment.run_multiple
        ```
    *   Most experiments also offer a test mode for a quicker run with reduced trials/subjects:
        ```bash
        python -m experiment.run_multiple --test
        ```
    *   For experiments that support different agent types (e.g., 'llm' vs 'random'), you might need to modify the `run_multiple.py` script or check for command-line arguments to switch between them, as noted in `task1-1_con/readme.md`.

5.  **Output Files:**
    *   Output data, logs, and conversation histories are typically saved in a `logs/` directory within each experiment folder (or as configured).
    *   Common outputs include:
        *   `.csv` files with trial-by-trial data.
        *   `.json` files with prompts and/or full conversation histories for LLM agents.
        *   `.log` files with detailed execution logs.

Please refer to the `readme.md` file within each specific task directory (`task2-1/readme.md`, `task2-2/readme.md`, etc.) for detailed instructions tailored to that particular experimental condition.

## Contact

hroyhong@gmail.com