# Clear workspace (optional, but good practice)
rm(list=ls())

# Load necessary libraries
library(dplyr)
library(DEoptim)
library(parallel)
library(snow)
library(SnowballC) # Although loaded, SnowballC isn't directly used here

# --- Load Data ---
# Make sure this path is correct for your system
dat_path <- '/Users/hroyhong/Desktop/llm0412/data/1.combined_r.csv'
if (!file.exists(dat_path)) {
  stop("Data file not found at: ", dat_path)
}
dat <- read.csv(dat_path)

# --- Function to fit the Relative Model for one participant ---
pa_relative <- function(li) {
  # Load libraries within the parallel function
  library(dplyr)
  library(DEoptim)
  
  # Load data again within the function (safer for parallel execution)
  dat_path_local <- '/Users/hroyhong/Desktop/llm0412/data/1.combined_r.csv'
  dat_local <- read.csv(dat_path_local)
  
  list_ID <- unique(dat_local$subj)
  # subj_count <- length(list_ID) # Not strictly needed inside LogL
  
  # Filter data for the current subject and relevant casinos
  dd <- dat_local[dat_local$subj == list_ID[li] & (dat_local$casino == 2 | dat_local$casino == 3), ]
  
  # --- Log-Likelihood Function for the Relative Model ---
  LogL_relative <- function(pp) {
    # Parameters:
    len1 <- pp[1] # Option learning rate (relative value)
    len2 <- pp[2] # Context learning rate
    tau <- pp[3]  # Inverse temperature
    
    logl <- 0 # Initialize log-likelihood
    
    # Initialize value vectors for each casino context
    # Each vector: c(value_option1, value_option2, value_context)
    # Initializing to 0, similar to the first example's first run
    value2 <- c(0, 0, 0)
    value3 <- c(0, 0, 0)
    
    for (i in 1:nrow(dd)) {
      choose <- dd$decision[i] # Participant's choice (1 or 2)
      reward1 <- dd$reward[i]  # Reward received for the chosen option
      casino <- dd$casino[i]   # Current casino context (2 or 3)
      
      # Select the appropriate value vector for the current casino
      if (casino == 2) {
        current_value <- value2
      } else { # casino == 3
        current_value <- value3
      }
      
      # --- Action Selection (Softmax) ---
      # Use only the option values (elements 1 and 2)
      val_option1 <- current_value[1]
      val_option2 <- current_value[2]
      
      # Robust softmax calculation to avoid Inf/Inf or exp(large number) issues
      max_val <- max(val_option1, val_option2)
      prob_option1 <- exp(tau * (val_option1 - max_val)) / (exp(tau * (val_option1 - max_val)) + exp(tau * (val_option2 - max_val)))
      prob_option2 <- 1.0 - prob_option1 # More stable than calculating separately
      
      # Get probability of the chosen option
      if (choose == 1) {
        prob_choose <- prob_option1
      } else {
        prob_choose <- prob_option2
      }
      
      # --- Log-Likelihood Calculation ---
      # Add small epsilon to prevent log(0)
      epsilon <- 1e-9
      logl <- logl + log(max(prob_choose, epsilon))
      
      # --- Value Updates (Relative Model Logic) ---
      # Identify the unchosen option (assuming only 2 options: 1 and 2)
      unchosen <- 3 - choose # If choose=1, unchosen=2; if choose=2, unchosen=1
      
      # Prediction Errors:
      # pe1: For the chosen option's relative value
      pe1 <- reward1 - current_value[3] - current_value[choose]
      # pe2: For the context value
      pe2 <- (reward1 + current_value[unchosen]) / 2 - current_value[3]
      
      # Update chosen option's relative value (element `choose`)
      current_value[choose] <- current_value[choose] + len1 * pe1
      
      # Update context value (element 3)
      current_value[3] <- current_value[3] + len2 * pe2
      
      # --- Store Updated Values Back ---
      # Write the updated 'current_value' back to the casino-specific vector
      if (casino == 2) {
        value2 <- current_value
      } else { # casino == 3
        value3 <- current_value
      }
    } # End of trial loop
    
    # --- Return Negative Log-Likelihood ---
    neg_logl <- -logl
    
    # Handle cases where optimization might lead to non-finite values
    if (!is.finite(neg_logl)) {
      neg_logl <- 1e6 # Return a large penalty value
    }
    return(neg_logl)
  } # End of LogL_relative function
  
  # --- Optimization using DEoptim ---
  # Parameters: len1 (option LR), len2 (context LR), tau
  # Bounds: Learning rates [0, 1], Tau [0, 20] (adjust tau upper bound if needed)
  estimates_relative = DEoptim(LogL_relative,
                               lower = c(0, 0, 0),
                               upper = c(1, 1, 20),
                               control = DEoptim.control(trace = FALSE, # Set to TRUE to see progress
                                                         NP = 80,    # Population size (>= 10 * number of parameters)
                                                         itermax = 250) # Max iterations
  )
  
  # --- Extract Results ---
  # Use summary to get the best parameters and value
  summary_est <- summary(estimates_relative)
  best_params <- summary_est$optim$bestmem
  best_negLL <- summary_est$optim$bestval
  
  # Create the final output vector
  final_result <- c(best_params[1], # len1 (option LR)
                    best_params[2], # len2 (context LR)
                    best_params[3], # tau
                    best_negLL,     # Minimized Negative Log-Likelihood
                    list_ID[li])    # Subject ID
  
  return(final_result)
} # End of pa_relative function


# --- Parallel Execution Setup ---
list_ID <- unique(dat$subj)
num_cores <- detectCores() - 1 # Use one less than total cores typically
if (num_cores < 1) num_cores <- 1 # Ensure at least one core
cl <- makeCluster(num_cores) # Use detected number of cores

# Export necessary variables/functions to the cluster if needed (often handled by parLapply)
# clusterExport(cl, varlist=c("dat_path")) # Usually not needed if data is loaded inside the function

# --- Run the Fitting in Parallel ---
# Use the new function name 'pa_relative'
results <- parLapply(cl, 1:length(list_ID), pa_relative)

# Stop the cluster
stopCluster(cl)

# --- Combine and Save Results ---
# Combine list of results into a data frame
res_df <- do.call('rbind', results)
res_df <- as.data.frame(res_df) # Convert matrix to data frame

# Assign meaningful column names
colnames(res_df) <- c('len_option', 'len_context', 'tau', 'NegLL', 'ID')

# Convert relevant columns to numeric (parLapply might return characters)
numeric_cols <- c('len_option', 'len_context', 'tau', 'NegLL')
for(col in numeric_cols) {
  res_df[, col] <- as.numeric(as.character(res_df[, col]))
}
res_df$ID <- as.character(res_df$ID) # Keep ID as character or factor

# Define output path and filename for the relative model results
output_path <- '/Users/hroyhong/Desktop/llm0412/task1/result/relative_model_result_r.csv'

# Save the results
write.table(res_df, output_path, row.names = FALSE, col.names = TRUE, sep = ",")

print(paste("Relative model fitting complete. Results saved to:", output_path))
print("Summary of estimated parameters:")
summary(res_df[, numeric_cols]) # Show summary stats of numeric results



# --- Plotting Script for Relative Model Results ---

# 1. Load necessary libraries
library(ggplot2) # For plotting
library(dplyr)   # For data manipulation (like filtering)
library(tidyr)   # For reshaping data (pivot_longer)

# 2. Define the path to your results file
results_file <- '/Users/hroyhong/Desktop/llm0412/task1/result/relative_model_result_r.csv' # Use the corrected file name

# 3. Load the results data
if (file.exists(results_file)) {
  res_df <- read.csv(results_file)
  print(paste("Successfully loaded results from:", results_file))
  print("First few rows of the data:")
  print(head(res_df))
  
  # Ensure correct data types (especially after reading CSV)
  numeric_cols <- c('len_option', 'len_context', 'tau', 'NegLL')
  suppressWarnings({ # Suppress warnings about NAs introduced by coercion
    for(col in numeric_cols) {
      if(col %in% names(res_df)) {
        res_df[, col] <- as.numeric(as.character(res_df[, col]))
      } else {
        warning(paste("Column", col, "not found in results file."))
      }
    }
  })
  if("ID" %in% names(res_df)) {
    res_df$ID <- as.character(res_df$ID)
  }
  
  
  # 4. Data Preparation for Plotting
  
  # Remove rows where fitting might have failed (containing NAs in parameters)
  res_clean <- res_df %>%
    filter(complete.cases(len_option, len_context, tau, NegLL)) # Keep only rows with valid numbers for all key params
  
  num_valid_participants <- nrow(res_clean)
  print(paste("Number of participants with valid results:", num_valid_participants))
  
  if (num_valid_participants > 0) {
    
    # Reshape data from wide to long format for easier plotting with facets
    res_long <- res_clean %>%
      select(ID, len_option, len_context, tau, NegLL) %>% # Select columns to plot
      pivot_longer(cols = -ID,               # Keep ID, pivot all other selected cols
                   names_to = "parameter",   # New column for parameter names
                   values_to = "value")      # New column for parameter values
    
    print("Data reshaped for plotting (first few rows):")
    print(head(res_long))
    
    # 5. Create Plots
    
    # --- Plot 1: Histograms ---
    plot_hist <- ggplot(res_long, aes(x = value)) +
      geom_histogram(bins = 15, fill = "skyblue", color = "black", alpha = 0.7) +
      # Create separate panels for each parameter
      # Use 'scales = "free_x"' so each parameter gets its own x-axis range
      facet_wrap(~ parameter, scales = "free_x") +
      labs(title = "Distribution of Estimated Parameters (Relative Model)",
           subtitle = paste("Based on", num_valid_participants, "participants"),
           x = "Parameter Value",
           y = "Frequency (Number of Participants)") +
      theme_bw() +
      theme(strip.background = element_rect(fill="grey90"),
            strip.text = element_text(face = "bold"),
            plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5))
    
    # --- Plot 2: Density Plots ---
    plot_density <- ggplot(res_long, aes(x = value)) +
      geom_density(fill = "lightgreen", color = "darkgreen", alpha = 0.7) +
      facet_wrap(~ parameter, scales = "free_x") + # Separate panels, free x-axis
      labs(title = "Density of Estimated Parameters (Relative Model)",
           subtitle = paste("Based on", num_valid_participants, "participants"),
           x = "Parameter Value",
           y = "Density") +
      theme_minimal() +
      theme(strip.text = element_text(face = "bold"),
            plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5))
    
    # --- Optional Plot 3: Box Plots ---
    plot_box <- ggplot(res_long, aes(x = parameter, y = value, fill = parameter)) +
      geom_boxplot(alpha = 0.8, show.legend = FALSE) + # Don't need legend if fill=parameter
      # Use 'scales = "free_y"' because parameters have different ranges
      facet_wrap(~ parameter, scales = "free_y") +
      labs(title = "Box Plots of Estimated Parameters (Relative Model)",
           subtitle = paste("Based on", num_valid_participants, "participants"),
           x = "Parameter", # X-axis is implicitly the parameter due to facetting
           y = "Parameter Value") +
      theme_bw() +
      theme(strip.text = element_blank(), # Remove facet titles if x-axis label is clear
            axis.text.x = element_blank(), # Remove x-axis text (redundant with facets)
            axis.ticks.x = element_blank(), # Remove x-axis ticks
            plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5))
    
    
    # 6. Display Plots
    print(plot_hist)
    print(plot_density)
    print(plot_box)
    
    
    # 7. Save Plots (Optional)
    plot_output_path <- '/Users/hroyhong/Desktop/llm0412/task1/result/' # Define output directory
    ggsave(paste0(plot_output_path, "relative_model_param_histograms.png"), plot_hist, width = 8, height = 6)
    ggsave(paste0(plot_output_path, "relative_model_param_density.png"), plot_density, width = 8, height = 6)
    ggsave(paste0(plot_output_path, "relative_model_param_boxplots.png"), plot_box, width = 8, height = 6)
    print(paste("Plots saved to:", plot_output_path))
    
  } else {
    print("No valid (non-NA) participant data found to plot.")
  }
  
} else {
  print(paste("Error: Results file not found at:", results_file))
  print("Please run the model fitting script first.")
}