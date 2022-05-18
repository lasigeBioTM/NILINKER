#install.packages('effsize')
#install.packages('rstatix')
#install.packages('ggpubr')
library(effsize) 
library(rstatix)
library(ggpubr)
library(dplyr)

#https://aclanthology.org/P18-1128.pdf
#https://stats.stackexchange.com/questions/73533/why-do-zero-differences-not-enter-computation-in-the-wilcoxon-signed-ranked-test
#https://www.statskingdom.com/175wilcoxon_signed_ranks.html

total_pred <- 447

model1_correct <- 378
model1_incorrect <- total_pred - model1_correct

model2_correct <- 385
model2_incorrect <- total_pred - model2_correct

# Correct predictions by both model1 and model2
both_corr = min(model1_correct, model2_correct)

# Incorrect predictions by both model1 and model2
both_incorr = min(model1_incorrect, model2_incorrect)

if (model1_correct >= model2_correct){
    # Model 1 has higher number of correct predictions
    model1_corr = model1_correct - model2_correct
    model2_corr = 0
} else {
    # Model 2 has higher number of correct predictions
    model2_corr = model2_correct - model1_correct
    model1_corr = 0
}

df <- data.frame(model1=c(1,1,0,0), model2=c(1,0,1,0), ntimes=c(both_corr, model1_corr, model2_corr,both_incorr))
df <- as.data.frame(lapply(df, rep, df$ntimes))

final_df <- df[c("model1", "model2")]

# See https://www.datanovia.com/en/lessons/wilcoxon-test-in-r/
# Transform into long data: 
# gather the before and after values in the same column
final_df.long <- final_df %>%
  gather(key = "group", value = "accuracy", model1, model2)

final_df.long %>%
  group_by(group) %>%
  get_summary_stats(accuracy, type = "median")

# Wilcoxon signed rank test on paired samples
stat.test <- final_df.long  %>%
  wilcox_test(accuracy ~ group, paired = TRUE) %>%
  add_significance()

stat.test

# Effect size
final_df.long  %>%
  wilcox_effsize(accuracy ~ group, paired = TRUE, alternative = "two.sided")