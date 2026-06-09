# ============================================================
# Random Forest + CRD + CRFD experiments for mlc_churn
# Author: Your Name
# Course: Design and Analysis of Experiments
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 0. Global settings
# ============================================================

SEED = 1234
DATA_PATH = "mlc_churn.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("===== DATA OVERVIEW =====")
print("Shape:", df.shape)
print(df.head())
print(df.info())

# Chuẩn hóa tên cột nếu cần
df.columns = [c.strip() for c in df.columns]

# Nếu có cột index hoặc cột không cần thiết thì loại
df = df.drop(columns=["index"], errors="ignore")

# Chuẩn hóa nhãn churn về dạng yes/no nếu cần
df["churn"] = df["churn"].astype(str).str.lower().str.strip()

print("\n===== CHURN DISTRIBUTION =====")
print(df["churn"].value_counts())
print(df["churn"].value_counts(normalize=True))


# ============================================================
# 2. Variable analysis and feature selection
# ============================================================

# Các biến charge thường gần như được tính trực tiếp từ minutes.
# Vì vậy loại để giảm dư thừa thông tin.
charge_cols = [
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge"
]

# Biến state là biến định danh nhiều mức.
# Ở phiên bản này loại state để mô hình gọn và tập trung vào hành vi sử dụng dịch vụ.
extra_drop_cols = ["state"]

drop_cols = charge_cols + extra_drop_cols

# Lưu bản mô tả biến bị loại
with open(os.path.join(OUTPUT_DIR, "removed_variables_explanation.txt"), "w", encoding="utf-8") as f:
    f.write("Removed variables:\n")
    f.write("- state: categorical nominal variable with many levels; removed to reduce dimensionality.\n")
    f.write("- total_*_charge: highly correlated with corresponding total_*_minutes; removed to reduce redundancy.\n")

# Phân tích tương quan trên các biến số trước khi loại
numeric_df = df.select_dtypes(include=[np.number])

corr = numeric_df.corr()
corr.to_csv(os.path.join(OUTPUT_DIR, "numeric_correlation_matrix.csv"))

print("\n===== HIGH CORRELATION PAIRS =====")
high_corr_pairs = []

cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        val = corr.iloc[i, j]
        if abs(val) >= 0.90:
            high_corr_pairs.append((cols[i], cols[j], val))

high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=["variable_1", "variable_2", "correlation"]
)

print(high_corr_df)
high_corr_df.to_csv(os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"), index=False)

# Vẽ heatmap tương quan bằng matplotlib
plt.figure(figsize=(12, 10))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation matrix of numeric variables")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# Loại biến
df_model = df.drop(columns=drop_cols, errors="ignore")

print("\n===== COLUMNS USED FOR MODEL =====")
print(df_model.columns.tolist())


# ============================================================
# 3. Prepare X, y and preprocessing pipeline
# ============================================================

X = df_model.drop(columns=["churn"])
y = df_model["churn"]

categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ],
    remainder="drop"
)


def build_rf_model(max_depth=None):
    """
    Build Random Forest model.
    Only max_depth is changed according to experiment design.
    Other hyperparameters use sklearn defaults.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("rf", RandomForestClassifier(
                max_depth=max_depth,
                random_state=SEED
            ))
        ]
    )
    return model


# ============================================================
# 4. Baseline model evaluation using repeated stratified k-fold
# ============================================================

print("\n===== BASELINE EVALUATION: k=10, repeat=10, max_depth=None =====")

baseline_cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=10,
    random_state=SEED
)

baseline_model = build_rf_model(max_depth=None)

# cross_val_predict để lấy dự đoán tổng thể
y_pred = cross_val_predict(
    baseline_model,
    X,
    y,
    cv=baseline_cv
)

baseline_f1 = f1_score(y, y_pred, pos_label="yes")

print("Baseline F1:", baseline_f1)
print("\nClassification report:")
print(classification_report(y, y_pred, labels=["no", "yes"]))

print("\nConfusion matrix:")
print(confusion_matrix(y, y_pred, labels=["no", "yes"]))

with open(os.path.join(OUTPUT_DIR, "baseline_model_evaluation.txt"), "w", encoding="utf-8") as f:
    f.write("Baseline model: Random Forest, max_depth=None, k=10, repeat=10\n")
    f.write(f"F1 positive class yes: {baseline_f1}\n\n")
    f.write("Classification report:\n")
    f.write(classification_report(y, y_pred, labels=["no", "yes"]))
    f.write("\nConfusion matrix labels [no, yes]:\n")
    f.write(str(confusion_matrix(y, y_pred, labels=["no", "yes"])))


# ============================================================
# 5. Function to run repeated stratified k-fold
# ============================================================

def run_repeated_cv(k, max_depth):
    """
    Run repeated stratified k-fold and return fold-level results.
    Each row is one fold result.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=k,
        n_repeats=10,
        random_state=SEED
    )

    rows = []

    for run_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        repeat_id = (run_id - 1) // k + 1
        fold_id = (run_id - 1) % k + 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_rf_model(max_depth=max_depth)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_test_pred, pos_label="yes")

        rows.append({
            "repeat": repeat_id,
            "fold": fold_id,
            "k": k,
            "max_depth": "None" if max_depth is None else str(max_depth),
            "f1": f1
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. CRD experiment
# Factor: k = 3, 5, 10
# Fixed: max_depth = None
# Response: F1-score
# ============================================================

print("\n===== RUNNING CRD EXPERIMENT =====")

crd_fold_results_list = []

for k in [3, 5, 10]:
    print(f"Running CRD: k={k}, max_depth=None")
    result = run_repeated_cv(k=k, max_depth=None)
    crd_fold_results_list.append(result)

crd_fold_results = pd.concat(crd_fold_results_list, ignore_index=True)

# Lưu kết quả từng fold nếu muốn kiểm tra chi tiết
crd_fold_results.to_csv(
    os.path.join(OUTPUT_DIR, "crd_fold_level_results.csv"),
    index=False
)

# Cách tốt hơn cho phân tích thiết kế thí nghiệm:
# lấy trung bình F1 theo từng repeat để mỗi mức k có đúng 10 quan sát.
crd_results = (
    crd_fold_results
    .groupby(["repeat", "k", "max_depth"], as_index=False)
    ["f1"]
    .mean()
)

crd_results.to_csv(
    os.path.join(OUTPUT_DIR, "crd_results.csv"),
    index=False
)

print("\nCRD repeat-level results:")
print(crd_results.head())
print(crd_results.groupby("k")["f1"].describe())


# ============================================================
# 7. CRFD experiment
# Factors:
#   k = 3, 5, 10
#   max_depth = 3, 5, None
# Response: F1-score
# ============================================================

print("\n===== RUNNING CRFD EXPERIMENT =====")

crfd_fold_results_list = []

for k in [3, 5, 10]:
    for max_depth in [3, 5, None]:
        print(f"Running CRFD: k={k}, max_depth={max_depth}")
        result = run_repeated_cv(k=k, max_depth=max_depth)
        crfd_fold_results_list.append(result)

crfd_fold_results = pd.concat(crfd_fold_results_list, ignore_index=True)

# Lưu kết quả từng fold
crfd_fold_results.to_csv(
    os.path.join(OUTPUT_DIR, "crfd_fold_level_results.csv"),
    index=False
)

# Lấy trung bình theo repeat để thiết kế cân bằng:
# 3 mức k × 3 mức max_depth × 10 repeat = 90 dòng
crfd_results = (
    crfd_fold_results
    .groupby(["repeat", "k", "max_depth"], as_index=False)
    ["f1"]
    .mean()
)

crfd_results.to_csv(
    os.path.join(OUTPUT_DIR, "crfd_results.csv"),
    index=False
)

print("\nCRFD repeat-level results:")
print(crfd_results.head())
print(crfd_results.groupby(["k", "max_depth"])["f1"].describe())


# ============================================================
# 8. Python summary statistics and plots
# ============================================================

def mean_ci_table(data, group_cols, value_col="f1"):
    """
    Compute mean and 95% CI using t distribution.
    """
    rows = []

    grouped = data.groupby(group_cols)

    for group_name, group_data in grouped:
        vals = group_data[value_col].values
        n = len(vals)
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1)
        se = sd / np.sqrt(n)

        # t critical xấp xỉ dùng 1.96 nếu không dùng scipy
        # Nếu muốn chính xác hơn, dùng scipy.stats.t.ppf
        try:
            from scipy.stats import t
            t_crit = t.ppf(0.975, df=n - 1)
        except Exception:
            t_crit = 1.96

        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se

        if not isinstance(group_name, tuple):
            group_name = (group_name,)

        row = {}
        for col, val in zip(group_cols, group_name):
            row[col] = val

        row.update({
            "n": n,
            "mean_f1": mean,
            "sd_f1": sd,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

        rows.append(row)

    return pd.DataFrame(rows)


crd_summary = mean_ci_table(crd_results, ["k"])
crd_summary.to_csv(os.path.join(OUTPUT_DIR, "crd_summary_mean_ci.csv"), index=False)

crfd_summary = mean_ci_table(crfd_results, ["k", "max_depth"])
crfd_summary.to_csv(os.path.join(OUTPUT_DIR, "crfd_summary_mean_ci.csv"), index=False)

print("\n===== CRD SUMMARY =====")
print(crd_summary)

print("\n===== CRFD SUMMARY =====")
print(crfd_summary)


# CRD plot
plt.figure(figsize=(7, 5))
plt.bar(crd_summary["k"].astype(str), crd_summary["mean_f1"])
plt.errorbar(
    crd_summary["k"].astype(str),
    crd_summary["mean_f1"],
    yerr=[
        crd_summary["mean_f1"] - crd_summary["ci_lower"],
        crd_summary["ci_upper"] - crd_summary["mean_f1"]
    ],
    fmt="none",
    capsize=5
)
plt.xlabel("k")
plt.ylabel("Mean F1-score")
plt.title("CRD: Effect of k on F1-score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crd_mean_ci_plot.png"), dpi=300)
plt.close()


# CRFD interaction plot
plt.figure(figsize=(8, 5))

for md in ["3", "5", "None"]:
    temp = crfd_summary[crfd_summary["max_depth"] == md].sort_values("k")
    plt.plot(
        temp["k"].astype(str),
        temp["mean_f1"],
        marker="o",
        label=f"max_depth={md}"
    )

plt.xlabel("k")
plt.ylabel("Mean F1-score")
plt.title("CRFD: Interaction plot of k and max_depth")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crfd_interaction_plot.png"), dpi=300)
plt.close()


print("\nAll outputs saved to:", OUTPUT_DIR)