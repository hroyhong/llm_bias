# --- 1. Define file path ---
file_path <- "/Users/hroyhong/Desktop/llm0412/task2/result/parallel_mle_result_complete_conf.csv"

# --- 2. Load Data ---
# Read the CSV, assuming the first row is the header
data <- read.csv(file_path, header = TRUE)

# --- 3. Descriptive Statistics ---
cat("--- Descriptive Statistics ---\n")

cat("\nSummary for par1:\n")
print(summary(data$par1)) # Provides min, 1st Qu, Median, Mean, 3rd Qu, Max

cat("\nSummary for par2:\n")
print(summary(data$par2))

cat("\nStandard Deviation for par1:", sd(data$par1), "\n")
cat("Standard Deviation for par2:", sd(data$par2), "\n")

cat("\n-----------------------------\n")

# --- 4. Perform Paired T-test ---
cat("\n--- Paired T-test ---\n")
# Conduct the paired t-test between par1 and par2
# The result object 't_test_result' holds t-statistic, p-value, etc.
t_test_result <- t.test(data$par1, data$par2, paired = TRUE)

# --- 5. Display T-test Results ---
# Print the standard output of the t-test
print(t_test_result)

# --- 6. Brief Interpretation ---
p_value <- t_test_result$p.value
alpha <- 0.05

cat("\n--- Brief Interpretation ---\n")
if (p_value < alpha) {
  cat(sprintf("P-value (%.4f) < %.2f: Reject H0. Statistically significant difference found.\n", p_value, alpha))
} else {
  cat(sprintf("P-value (%.4f) >= %.2f: Fail to reject H0. No statistically significant difference found.\n", p_value, alpha))
}
cat("---------------------------\n")
