# plot_llm_params.R
# -----------------

# Load required libraries
library(ggplot2)

# 1. Read CSV output from the MLE fitting
#    Make sure the path matches where your CSV file is actually saved.
df <- read.csv("/Users/hroyhong/Desktop/llm0412/task1/result/parallel_mle_result_r.csv", header = TRUE)
head(df)
summary(df)


# If your CSV does not already have clear column names, rename them here:
#   a = alpha_plus, b = alpha_minus, c = tau, d = bestval, e = subj
# If your existing CSV already has good names, adjust accordingly.
colnames(df) <- c("alpha_plus", "alpha_minus", "tau", "bestval", "subj")

# 2. Compute means and standard errors across all subjects
alpha_plus_mean  <- mean(df$alpha_plus)
alpha_plus_se    <- sd(df$alpha_plus) / sqrt(nrow(df))
alpha_minus_mean <- mean(df$alpha_minus)
alpha_minus_se   <- sd(df$alpha_minus) / sqrt(nrow(df))


# 3. Organize data for plotting
plot_df <- data.frame(
  Condition = factor(c("alpha^+", "alpha^-"), levels = c("alpha^+", "alpha^-")),
  Mean      = c(alpha_plus_mean, alpha_minus_mean),
  SE        = c(alpha_plus_se, alpha_minus_se)
)


# 4. Create the bar plot with error bars
p <- ggplot(plot_df, aes(x = Condition, y = Mean, fill = Condition)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.2) +
  scale_fill_manual(values = c("orchid4", "orchid4")) +
  labs(x = NULL, y = "Learning rate") +
  theme_classic() +
  theme(legend.position = "none") +
  # Use expressions to get the superscript plus/minus in the axis labels
  scale_x_discrete(labels = c(expression(alpha^"+"), expression(alpha^"-")))



# 5. Print the plot to your R graphics window
print(p)

# (Optional) Save the plot to a file
ggsave("/Users/hroyhong/Desktop/llm0412/task1/result/llm_learning_rates_r.png", plot = p, width = 4, height = 4, dpi = 300)
