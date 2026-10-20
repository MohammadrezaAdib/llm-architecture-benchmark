# ==============================================================================
# Filename: statistical_analysis.R
# Description: Linear Mixed-Effects Model (LMM) for LLM Benchmark Data.
# Requires: install.packages(c("lme4", "lmerTest", "emmeans", "dplyr", "readr"))
# ==============================================================================

library(lme4)
library(lmerTest)
library(emmeans)
library(dplyr)
library(readr)

# 1. Load data
# Ensure data.csv is in the working directory
df <- read_csv("data.csv", show_col_types = FALSE)

# Convert to factors for LMM
df$model <- as.factor(df$model)
df$prompt <- as.factor(df$prompt)
df$run <- as.factor(df$run)

# 2. Descriptive Statistics
desc <- df %>% 
  group_by(model) %>% 
  summarise(
    mean_composite = round(mean(composite), 2), 
    sd_composite = round(sd(composite), 2),
    mean_ses = round(mean(ses), 2),
    mean_osc = round(mean(osc), 2),
    .groups = "drop"
  )
print("=== Descriptive Stats ===")
print(desc)

# 3. Linear Mixed-Effects Model (LMM)
# Composite score is predicted by model, with random intercepts for prompt
m_lmm <- lmer(composite ~ model + (1 | prompt), data = df)

print("=== ANOVA (Main Effect of Model) ===")
print(anova(m_lmm))

# 4. Post-hoc Pairwise Comparisons
# Using Tukey's HSD with Bonferroni correction
emm <- emmeans(m_lmm, pairwise ~ model, adjust = "bonferroni")
print("=== Post-hoc Comparisons ===")
print(emm$contrasts)
