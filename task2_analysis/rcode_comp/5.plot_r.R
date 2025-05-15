# plot_llm_params_chosen_unchosen.R
# ---------------------------------

# Load required libraries
library(ggplot2)
library(dplyr) # Using dplyr for easier calculation

# 1. Read CSV output from the *new* MLE fitting (5 parameters)
#    Make sure the path matches where your CSV file is actually saved.
#    IMPORTANT: Ensure this is the file generated AFTER modifying the MLE script for 5 parameters.
file_path <- "/Users/hroyhong/Desktop/llm0412/task1/result/parallel_mle_result_r_comp.csv"
if (!file.exists(file_path)) {
  stop("Error: CSV file not found at specified path: ", file_path)
}
df <- read.csv(file_path, header = TRUE)

# Check if the dataframe looks reasonable (should have more columns than before)
print("Original Column Names:")
print(colnames(df))
print("First few rows:")
print(head(df))
print("Summary:")
print(summary(df))

# Check if the number of columns seems correct for 5 parameters + logL + subjID
# Expecting potentially 7 or more columns depending on how write.table formatted empty ones.
# We will select the first 7 assuming they are the relevant ones.
if (ncol(df) < 7) {
  stop("Error: CSV file does not appear to have enough columns for the 5-parameter model results.")
}
df <- df[, 1:7] # Select the first 7 columns


# 2. Rename columns to be descriptive for the 5-parameter model
#    **CRITICAL STEP**: Adjust this order if your MLE script saved them differently.
colnames(df) <- c("alpha_pos_c", "alpha_neg_c", "alpha_pos_u", "alpha_neg_u", "tau", "bestval", "subj")

print("Renamed Columns Head:")
print(head(df))

# 3. Calculate means and standard errors for all four learning rates
#    Using dplyr for clarity
summary_stats <- df %>%
  summarise(
    alpha_pos_c_mean = mean(alpha_pos_c, na.rm = TRUE),
    alpha_pos_c_se   = sd(alpha_pos_c, na.rm = TRUE) / sqrt(n()),
    alpha_neg_c_mean = mean(alpha_neg_c, na.rm = TRUE),
    alpha_neg_c_se   = sd(alpha_neg_c, na.rm = TRUE) / sqrt(n()),
    alpha_pos_u_mean = mean(alpha_pos_u, na.rm = TRUE),
    alpha_pos_u_se   = sd(alpha_pos_u, na.rm = TRUE) / sqrt(n()),
    alpha_neg_u_mean = mean(alpha_neg_u, na.rm = TRUE),
    alpha_neg_u_se   = sd(alpha_neg_u, na.rm = TRUE) / sqrt(n())
  )


# 4. Organize data for plotting (for the example image style)
# Create a single factor representing the four distinct conditions in the desired order
plot_df <- data.frame(
  # Define the condition names and ensure they are ordered correctly for the plot
  Condition = factor(
    c("Chosen_Alpha_Pos", "Chosen_Alpha_Neg", "Unchosen_Alpha_Pos", "Unchosen_Alpha_Neg"),
    levels = c("Chosen_Alpha_Pos", "Chosen_Alpha_Neg", "Unchosen_Alpha_Pos", "Unchosen_Alpha_Neg")
  ),
  # Assign the corresponding means calculated earlier
  Mean      = c(summary_stats$alpha_pos_c_mean, summary_stats$alpha_neg_c_mean,
                summary_stats$alpha_pos_u_mean, summary_stats$alpha_neg_u_mean),
  # Assign the corresponding standard errors
  SE        = c(summary_stats$alpha_pos_c_se, summary_stats$alpha_neg_c_se,
                summary_stats$alpha_pos_u_se, summary_stats$alpha_neg_u_se)
)

print("Data frame for plotting (New Structure):")
print(plot_df)

# 5. Create the bar plot like the example image

# Choose a single color similar to the example (adjust color name as needed)
plot_color <- "palevioletred3"

p <- ggplot(plot_df, aes(x = Condition, y = Mean)) + # Base mapping: x and y
  geom_bar(stat = "identity", fill = plot_color, width = 0.7) + # Bars with fixed color
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.25) + # Error bars
  # Set the specific labels for the x-axis ticks corresponding to the Condition levels
  scale_x_discrete(labels = c(
    "Chosen_Alpha_Pos"   = expression(alpha^"+"),
    "Chosen_Alpha_Neg"   = expression(alpha^"-"),
    "Unchosen_Alpha_Pos" = expression(alpha^"+"),
    "Unchosen_Alpha_Neg" = expression(alpha^"-")
  )) +
  labs(x = NULL, y = "Learning rate") + # Set y-axis label, remove x-axis label text
  theme_classic(base_size = 16) + # Use classic theme, slightly larger text
  theme(
    axis.title.x = element_blank(),  # Remove space for x-axis title
    axis.text.x = element_text(size=14), # Increase size of alpha labels
    axis.ticks.x = element_blank()   # Remove x-axis ticks if desired (like example)
  )
# Optional: Add a title if desired


# Note: Adding the overarching "Chosen" / "Unchosen" labels with brackets below the axis
# requires more advanced techniques like annotations or patchwork, which go beyond
# minimal changes to the standard ggplot call. This code produces the bars and alpha labels.

print(p)
