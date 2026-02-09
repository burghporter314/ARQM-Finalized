import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

# Load each rater's Excel file
rater1_df = pd.read_excel("app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx")
rater2_df = pd.read_excel("app/util/validation/datasets/requirement_quality/reviewer_5/dataset_5_quality_unlabeled.xlsx")

column = "verifiability_violation"
# column = "singularity_violation"

# Compute Cohen's kappa
kappa = cohen_kappa_score(rater1_df[column], rater2_df[column])

print(f"Cohen's kappa: {kappa:.3f}")

print(classification_report(rater1_df[column], rater2_df[column]))