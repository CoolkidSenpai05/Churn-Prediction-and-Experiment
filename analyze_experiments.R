# ============================================================
# Statistical analysis for CRD and CRFD experiments
# Course: Design and Analysis of Experiments
# Dataset: mlc_churn
# ============================================================

# ============================================================
# 0. Install and load packages
# ============================================================

packages <- c("car", "ggplot2", "dplyr")

for (p in packages) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# ============================================================
# 1. Paths and data loading
# ============================================================

OUTPUT_DIR <- "outputs"

crd_path <- file.path(OUTPUT_DIR, "crd_results.csv")
crfd_path <- file.path(OUTPUT_DIR, "crfd_results.csv")

if (!file.exists(crd_path)) {
  stop("Cannot find crd_results.csv in outputs folder.")
}

if (!file.exists(crfd_path)) {
  stop("Cannot find crfd_results.csv in outputs folder.")
}

crd <- read.csv(crd_path)
crfd <- read.csv(crfd_path)

# Convert experimental factors to factor type
crd$k <- as.factor(crd$k)

crfd$k <- as.factor(crfd$k)
crfd$max_depth <- as.factor(crfd$max_depth)

# Set readable order for max_depth
crfd$max_depth <- factor(
  crfd$max_depth,
  levels = c("3", "5", "None")
)

# Output file for statistical results
sink(file.path(OUTPUT_DIR, "statistical_analysis_outputs.txt"))

cat("============================================================\n")
cat("STATISTICAL ANALYSIS OUTPUTS\n")
cat("============================================================\n\n")


# ============================================================
# 2. Helper function: mean and 95% confidence interval
# ============================================================

mean_ci <- function(data, group_vars, response_var = "f1") {
  data %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      n = n(),
      mean_f1 = mean(.data[[response_var]], na.rm = TRUE),
      sd_f1 = sd(.data[[response_var]], na.rm = TRUE),
      se = sd_f1 / sqrt(n),
      t_value = qt(0.975, df = n - 1),
      ci_lower = mean_f1 - t_value * se,
      ci_upper = mean_f1 + t_value * se,
      .groups = "drop"
    )
}


# ============================================================
# 3. CRD analysis
# Factor: k = 3, 5, 10
# Response: F1-score
# ============================================================

cat("============================================================\n")
cat("CRD ANALYSIS\n")
cat("============================================================\n\n")

cat("CRD data structure:\n")
print(str(crd))

cat("\nCRD first rows:\n")
print(head(crd))

cat("\nCRD group sizes:\n")
print(table(crd$k))


# ------------------------------------------------------------
# 3.1 Mean and 95% confidence interval
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: Mean and 95% confidence interval\n")
cat("------------------------------------------------------------\n")

crd_summary <- mean_ci(crd, group_vars = c("k"), response_var = "f1")

print(crd_summary)

write.csv(
  crd_summary,
  file.path(OUTPUT_DIR, "r_crd_summary_mean_ci.csv"),
  row.names = FALSE
)


# ------------------------------------------------------------
# 3.2 Levene test for equality of variances
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: Levene test for equality of variances\n")
cat("H0: Variances of F1 are equal across k groups\n")
cat("------------------------------------------------------------\n")

levene_crd <- leveneTest(f1 ~ k, data = crd)
print(levene_crd)


# ------------------------------------------------------------
# 3.3 Linear model and ANOVA
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: Linear model lm(f1 ~ k)\n")
cat("H0: Mean F1 values are equal across k groups\n")
cat("------------------------------------------------------------\n")

model_crd <- lm(f1 ~ k, data = crd)

cat("\nSummary of CRD linear model:\n")
print(summary(model_crd))

cat("\nANOVA table for CRD linear model:\n")
print(anova(model_crd))


# ------------------------------------------------------------
# 3.4 ANOVA using aov()
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: ANOVA using aov(f1 ~ k)\n")
cat("------------------------------------------------------------\n")

aov_crd <- aov(f1 ~ k, data = crd)
print(summary(aov_crd))


# ------------------------------------------------------------
# 3.5 Residual normality check
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: Residual normality check\n")
cat("H0: Residuals are normally distributed\n")
cat("------------------------------------------------------------\n")

res_crd <- residuals(model_crd)
print(shapiro.test(res_crd))


# ------------------------------------------------------------
# 3.6 TukeyHSD pairwise comparison
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRD: TukeyHSD pairwise comparison\n")
cat("------------------------------------------------------------\n")

tukey_crd <- TukeyHSD(aov_crd)
print(tukey_crd)


# ============================================================
# 4. CRFD analysis
# Factors:
#   k = 3, 5, 10
#   max_depth = 3, 5, None
# Response: F1-score
# ============================================================

cat("\n\n============================================================\n")
cat("CRFD ANALYSIS\n")
cat("============================================================\n\n")

cat("CRFD data structure:\n")
print(str(crfd))

cat("\nCRFD first rows:\n")
print(head(crfd))

cat("\nCRFD group sizes:\n")
print(table(crfd$k, crfd$max_depth))


# ------------------------------------------------------------
# 4.1 Mean and 95% confidence interval
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: Mean and 95% confidence interval\n")
cat("------------------------------------------------------------\n")

crfd_summary <- mean_ci(
  crfd,
  group_vars = c("k", "max_depth"),
  response_var = "f1"
)

print(crfd_summary)

write.csv(
  crfd_summary,
  file.path(OUTPUT_DIR, "r_crfd_summary_mean_ci.csv"),
  row.names = FALSE
)


# ------------------------------------------------------------
# 4.2 Main-effects model
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: Main-effects model lm(f1 ~ k + max_depth)\n")
cat("------------------------------------------------------------\n")

model_crfd_main <- lm(f1 ~ k + max_depth, data = crfd)

cat("\nSummary of CRFD main-effects model:\n")
print(summary(model_crfd_main))

cat("\nANOVA table for CRFD main-effects model:\n")
print(anova(model_crfd_main))


# ------------------------------------------------------------
# 4.3 Interaction model
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: Interaction model lm(f1 ~ k * max_depth)\n")
cat("This model includes k, max_depth, and k:max_depth interaction\n")
cat("------------------------------------------------------------\n")

model_crfd_interaction <- lm(f1 ~ k * max_depth, data = crfd)

cat("\nSummary of CRFD interaction model:\n")
print(summary(model_crfd_interaction))

cat("\nANOVA table for CRFD interaction model:\n")
print(anova(model_crfd_interaction))


# ------------------------------------------------------------
# 4.4 ANOVA using aov()
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: ANOVA using aov(f1 ~ k * max_depth)\n")
cat("------------------------------------------------------------\n")

aov_crfd <- aov(f1 ~ k * max_depth, data = crfd)
print(summary(aov_crfd))


# ------------------------------------------------------------
# 4.5 Levene test across treatment combinations
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: Levene test across treatment combinations\n")
cat("H0: Variances are equal across k:max_depth groups\n")
cat("------------------------------------------------------------\n")

crfd$group <- interaction(crfd$k, crfd$max_depth)

levene_crfd <- leveneTest(f1 ~ group, data = crfd)
print(levene_crfd)


# ------------------------------------------------------------
# 4.6 Residual normality check
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: Residual normality check\n")
cat("H0: Residuals are normally distributed\n")
cat("------------------------------------------------------------\n")

res_crfd <- residuals(model_crfd_interaction)
print(shapiro.test(res_crfd))


# ------------------------------------------------------------
# 4.7 TukeyHSD for main factors and treatment combinations
# ------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("CRFD: TukeyHSD for factor k\n")
cat("------------------------------------------------------------\n")

print(TukeyHSD(aov_crfd, "k"))

cat("\n------------------------------------------------------------\n")
cat("CRFD: TukeyHSD for factor max_depth\n")
cat("------------------------------------------------------------\n")

print(TukeyHSD(aov_crfd, "max_depth"))

cat("\n------------------------------------------------------------\n")
cat("CRFD: TukeyHSD for all k:max_depth treatment combinations\n")
cat("------------------------------------------------------------\n")

aov_crfd_group <- aov(f1 ~ group, data = crfd)
tukey_crfd_group <- TukeyHSD(aov_crfd_group)
print(tukey_crfd_group)


sink()


# ============================================================
# 5. Plots
# ============================================================

# ------------------------------------------------------------
# 5.1 CRD mean + 95% CI plot
# ------------------------------------------------------------

p_crd_mean_ci <- ggplot(crd_summary, aes(x = k, y = mean_f1)) +
  geom_col(width = 0.6) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.15
  ) +
  labs(
    title = "CRD: Effect of k on F1-score",
    x = "Number of folds k",
    y = "Mean F1-score with 95% CI"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(OUTPUT_DIR, "r_crd_mean_ci_plot.png"),
  plot = p_crd_mean_ci,
  width = 7,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 5.2 CRD boxplot
# ------------------------------------------------------------

p_crd_box <- ggplot(crd, aes(x = k, y = f1)) +
  geom_boxplot() +
  geom_jitter(width = 0.08, alpha = 0.6) +
  labs(
    title = "CRD: Distribution of F1-score by k",
    x = "Number of folds k",
    y = "F1-score"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(OUTPUT_DIR, "r_crd_boxplot.png"),
  plot = p_crd_box,
  width = 7,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 5.3 CRD TukeyHSD plot
# ------------------------------------------------------------

# png(
#   filename = file.path(OUTPUT_DIR, "r_crd_tukey_plot.png"),
#   width = 900,
#   height = 600
# )
# plot(tukey_crd)
# dev.off()
# ------------------------------------------------------------
# CRD TukeyHSD plot using ggplot2
# ------------------------------------------------------------

tukey_crd_df <- as.data.frame(tukey_crd$k)
tukey_crd_df$comparison <- rownames(tukey_crd_df)

p_crd_tukey <- ggplot(
  tukey_crd_df,
  aes(x = comparison, y = diff)
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = lwr, ymax = upr),
    width = 0.15,
    linewidth = 0.8
  ) +
  coord_flip() +
  labs(
    title = "CRD: TukeyHSD pairwise comparisons",
    subtitle = "Differences in mean F1-score between k groups",
    x = "Pairwise comparison",
    y = "Difference in mean F1-score"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(OUTPUT_DIR, "r_crd_tukey_plot.png"),
  plot = p_crd_tukey,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 5.4 CRFD interaction plot
# ------------------------------------------------------------

# p_crfd_interaction <- ggplot(
#   crfd_summary,
#   aes(x = k, y = mean_f1, group = max_depth, color = max_depth)
# ) +
#   geom_line(linewidth = 1) +
#   geom_point(size = 2) +
#   geom_errorbar(
#     aes(ymin = ci_lower, ymax = ci_upper),
#     width = 0.08
#   ) +
#   labs(
#     title = "CRFD: Interaction between k and max_depth",
#     x = "Number of folds k",
#     y = "Mean F1-score with 95% CI",
#     color = "max_depth"
#   ) +
#   theme_minimal()

# crfd_summary$k_num <- as.numeric(as.character(crfd_summary$k))

# p_crfd_interaction <- ggplot(
#   crfd_summary,
#   aes(x = k_num, y = mean_f1, group = max_depth, color = max_depth)
# ) +
#   geom_line(linewidth = 1) +
#   geom_point(size = 2) +
#   geom_errorbar(
#     aes(ymin = ci_lower, ymax = ci_upper),
#     width = 0.15
#   ) +
#   scale_x_continuous(
#     breaks = c(3, 5, 10)
#   ) +
#   labs(
#     title = "CRFD: Interaction between k and max_depth",
#     x = "Number of folds k",
#     y = "Mean F1-score with 95% CI",
#     color = "max_depth"
#   ) +
#   theme_minimal()

# ggsave(
#   filename = file.path(OUTPUT_DIR, "r_crfd_interaction_plot.png"),
#   plot = p_crfd_interaction,
#   width = 8,
#   height = 5,
#   dpi = 300
# )

# Tạo biến k dạng numeric chỉ để vẽ
crfd_summary$k_num <- as.numeric(as.character(crfd_summary$k))

p_crfd_interaction <- ggplot(
  crfd_summary,
  aes(
    x = k_num,
    y = mean_f1,
    group = max_depth,
    color = max_depth,
    linetype = max_depth,
    shape = max_depth
  )
) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.18,
    linewidth = 0.7
  ) +
  scale_x_continuous(
    breaks = c(3, 5, 10),
    limits = c(2.7, 10.3)
  ) +
  scale_y_continuous(
    limits = c(0, 0.85),
    breaks = seq(0, 0.8, by = 0.1)
  ) +
  labs(
    title = "Interaction effect of k and max_depth on F1-score",
    subtitle = "Mean F1-score with 95% confidence intervals",
    x = "Number of folds (k)",
    y = "Mean F1-score",
    color = "max_depth",
    linetype = "max_depth",
    shape = "max_depth"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(OUTPUT_DIR, "r_crfd_interaction_plot.png"),
  plot = p_crfd_interaction,
  width = 8,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 5.5 CRFD boxplot by k and max_depth
# ------------------------------------------------------------

p_crfd_box <- ggplot(
  crfd,
  aes(x = max_depth, y = f1)
) +
  geom_boxplot() +
  facet_wrap(~ k) +
  labs(
    title = "CRFD: F1-score by max_depth and k",
    x = "max_depth",
    y = "F1-score"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(OUTPUT_DIR, "r_crfd_boxplot.png"),
  plot = p_crfd_box,
  width = 9,
  height = 5,
  dpi = 300
)


# ------------------------------------------------------------
# 5.6 CRFD TukeyHSD plot for treatment combinations
# ------------------------------------------------------------

png(
  filename = file.path(OUTPUT_DIR, "r_crfd_tukey_group_plot.png"),
  width = 1100,
  height = 800
)
plot(tukey_crfd_group)
dev.off()


# ------------------------------------------------------------
# 5.7 Residual diagnostic QQ plots
# ------------------------------------------------------------

png(
  filename = file.path(OUTPUT_DIR, "r_crd_qqplot.png"),
  width = 800,
  height = 600
)
qqnorm(res_crd, main = "CRD residual QQ plot")
qqline(res_crd)
dev.off()

png(
  filename = file.path(OUTPUT_DIR, "r_crfd_qqplot.png"),
  width = 800,
  height = 600
)
qqnorm(res_crfd, main = "CRFD residual QQ plot")
qqline(res_crfd)
dev.off()


# ============================================================
# 6. Final message
# ============================================================

cat("Analysis completed. Results saved in outputs folder.\n")