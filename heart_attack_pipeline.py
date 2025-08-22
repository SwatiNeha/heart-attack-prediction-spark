from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import random
random.seed(42)
np.random.seed(42)
from matplotlib.backends.backend_pdf import PdfPages

def remove_outliers(df, numeric_cols):
    for col_name in numeric_cols:
        q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df.filter((col(col_name) >= lower) & (col(col_name) <= upper))
    return df

def plot_imbalanced_class(df, target_col, filename):
    pd_df = df.select(target_col).toPandas()
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=pd_df, palette="Set2")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_correlation_matrix(df, numeric_cols, filename):
    pd_df = df.select(numeric_cols).toPandas()
    corr = pd_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_histograms_and_boxplots(df, numeric_cols, filename):
    pd_df = df.select(numeric_cols).toPandas()
    with PdfPages(filename) as pdf:
        for col_name in numeric_cols:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(pd_df[col_name], kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title(f'Histogram of {col_name}')
            sns.boxplot(x=pd_df[col_name], ax=axs[1], color='lightgreen')
            axs[1].set_title(f'Boxplot of {col_name}')
            pdf.savefig(fig)
            plt.close()

def compute_auc(df, label_col="HadHeartAttack_Index", probability_col="probability"):
    paired = df.select(label_col, probability_col).rdd.map(lambda row: (row[label_col], row[probability_col][1])).collect()
    y_true, y_prob = zip(*paired)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return roc_auc_score(y_true, y_prob)


def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_results(filename, model, train_accuracy, test_accuracy, running_time, feature_columns, auc_train, auc_test):
    coefficients = model.coefficients.toArray()
    intercept = model.intercept
    with open(filename, "w") as f:
        f.write(f"Intercept: {intercept}\n")
        for feature, coef in zip(feature_columns, coefficients):
            f.write(f"Feature: {feature}, Coefficient: {coef}\n")
        f.write(f"Train Accuracy: {train_accuracy}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Train AUC: {auc_train}\n")
        f.write(f"Test AUC: {auc_test}\n")
        f.write(f"Running time: {running_time} seconds\n")

def replace_values(df):
    df = df.withColumn("HadDiabetes", when(col("HadDiabetes") == "No, pre-diabetes or borderline diabetes", "Borderline")
        .when(col("HadDiabetes") == "Yes, but only during pregnancy (female)", "During Pregnancy").otherwise(col("HadDiabetes")))
    df = df.withColumn("SmokerStatus", when(col("SmokerStatus") == "Current smoker - now smokes some days", "Current smoker(Some days)")
        .when(col("SmokerStatus") == "Current smoker - now smokes every day", "Current smoker(Every day)").otherwise(col("SmokerStatus")))
    df = df.withColumn("ECigaretteUsage", when(col("ECigaretteUsage") == "Not at all (right now)", "Not at all")
        .when(col("ECigaretteUsage") == "Never used e-cigarettes in my entire life", "Never")
        .when(col("ECigaretteUsage") == "Use them every day", "Everyday")
        .when(col("ECigaretteUsage") == "Use them some days", "Somedays").otherwise(col("ECigaretteUsage")))
    df = df.withColumn("RaceEthnicityCategory", when(col("RaceEthnicityCategory") == "White only, Non-Hispanic", "White")
        .when(col("RaceEthnicityCategory") == "Black only, Non-Hispanic", "Black")
        .when(col("RaceEthnicityCategory") == "Other race only, Non-Hispanic", "Other Race")
        .when(col("RaceEthnicityCategory") == "Multiracial, Non-Hispanic", "Multi Racial").otherwise(col("RaceEthnicityCategory")))
    df = df.withColumn("CovidPos", when(col("CovidPos") == "Tested positive using home test without a health professional", "Yes")
        .otherwise(col("CovidPos")))
    return df

def plot_roc_curve(y_true, y_prob, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.2f})")

# Spark session
spark = SparkSession.builder.appName("HeartDiseasePipeline").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load and preprocess data
print("\n--- Reading Data ---")
df = spark.read.csv("hdfs:///user/swatiswat/inputs/heart_2022_with_nans.csv", header=True, inferSchema=True)
df.printSchema()
print(f"Original dataset has {df.count()} rows and {len(df.columns)} columns.")

df = df.dropna()
print(f"The dataset now contains {df.count()} rows and {len(df.columns)} columns after dropping NA's")

numeric_cols = [col for col, dtype in df.dtypes if dtype in ["int", "double"] and col != "HadHeartAttack"]
plot_correlation_matrix(df, numeric_cols, "correlation_matrix.pdf")
df = replace_values(df).drop("State", "WeightInKilograms", "RemovedTeeth", "FluVaxLast12", "PneumoVaxEver", "TetanusLast10Tdap").dropDuplicates()

numeric_cols = [col for col, dtype in df.dtypes if dtype in ["int", "double"] and col != "HadHeartAttack"]

plot_histograms_and_boxplots(df, numeric_cols, "numeric_features_distribution.pdf")

# Remove outliers
print("\n--- Removing Outliers from Numeric Columns ---")
df = remove_outliers(df, numeric_cols)
print(f"Final dataset after outlier removal has {df.count()} rows and {len(df.columns)} columns.")

df.drop_duplicates()

print(f"Dataset after duplicates removal has {df.count()} rows and {len(df.columns)} columns.")

plot_imbalanced_class(df, "HadHeartAttack", "imbalanced_classes.pdf")

counts = df.groupBy("HadHeartAttack").count().collect()
class_counts = {row["HadHeartAttack"]: row["count"] for row in counts}
majority_class = max(class_counts, key=class_counts.get)
minority_class = min(class_counts, key=class_counts.get)

# Compute imbalance ratio
imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
print("Imbalance ratio:", imbalance_ratio)

minority_df = df.filter(col("HadHeartAttack") == minority_class)
majority_df = df.filter(col("HadHeartAttack") == majority_class)
minority_count = minority_df.count()
majority_count = majority_df.count()
print(f"Minority class count (Yes): {minority_count}")
print(f"Majority class count (No): {majority_count}")

replication_factor = majority_count // minority_count
remainder = majority_count % minority_count

oversampled_minority_df = minority_df
for _ in range(replication_factor - 1):
    oversampled_minority_df = oversampled_minority_df.union(minority_df)

if remainder > 0:
    extra_minority_rows = minority_df.sample(withReplacement=True, fraction=remainder / minority_count, seed=42)
    oversampled_minority_df = oversampled_minority_df.union(extra_minority_rows)

balanced_df = majority_df.union(oversampled_minority_df).orderBy(rand())

balanced_df.groupBy("HadHeartAttack").count().show()

df = balanced_df

plot_imbalanced_class(df, "HadHeartAttack", "balanced_classes.pdf")


# Index categorical columns
categorical_cols = [ 'Sex', 'GeneralHealth', 'PhysicalActivities', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
    'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
    'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage',
    'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 'CovidPos', 'LastCheckupTime',
    'ChestScan', 'HIVTesting', 'HighRiskLastYear' ]
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_cols]
assembler = VectorAssembler(inputCols=[f"{col}_index" for col in categorical_cols] + numeric_cols, outputCol="features")
index_pipeline = Pipeline(stages=indexers)
df = index_pipeline.fit(df).transform(df)
df = StringIndexer(inputCol="HadHeartAttack", outputCol="HadHeartAttack_Index", handleInvalid="skip").fit(df).transform(df)


# Ready for splitting & pipeline
paramGrid = ParamGridBuilder().addGrid(LogisticRegression().regParam, [0.1, 0.01]).addGrid(LogisticRegression().elasticNetParam, [0.0, 0.5, 1.0]).addGrid(LogisticRegression().maxIter, [10, 50, 100]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="HadHeartAttack_Index", predictionCol="prediction", metricName="accuracy")

# ---------- LR without Scaling
print("\n--- Starting Logistic Regression Without Scaling Pipeline ---")
feature_columns=[f"{col}_index" for col in categorical_cols] + numeric_cols
train, test = df.randomSplit([0.8, 0.2], seed=42)
pipeline_basic = Pipeline(stages=[assembler, LogisticRegression(family="binomial",labelCol="HadHeartAttack_Index", featuresCol="features")])
cv_basic = CrossValidator(estimator=pipeline_basic, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
start_basic = time.time()
model_basic = cv_basic.fit(train)
elapsed_basic = time.time() - start_basic
train_pred_basic = model_basic.transform(train)
test_pred_basic = model_basic.transform(test)
train_acc_basic = evaluator.evaluate(train_pred_basic)
test_acc_basic = evaluator.evaluate(test_pred_basic)
auc_train_basic = compute_auc(train_pred_basic)
auc_test_basic = compute_auc(test_pred_basic)
print(f"BASIC - Train Acc: {train_acc_basic}, Test Acc: {test_acc_basic}, Train AUC: {auc_train_basic}, Test AUC: {auc_test_basic}, Time: {elapsed_basic}")
save_results("basic_logistic_regression.txt", model_basic.bestModel.stages[-1], train_acc_basic, test_acc_basic, elapsed_basic,feature_columns, auc_train_basic, auc_test_basic)
paired = test_pred_basic.select("HadHeartAttack_Index", "prediction", "probability").rdd \
    .filter(lambda row: row["prediction"] is not None) \
    .map(lambda row: (row["HadHeartAttack_Index"], row["prediction"], row["probability"][1])) \
    .collect()
y_true, y_pred, y_prob = zip(*paired)
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)
plot_confusion_matrix(y_true, y_pred, "confusion_matrix_basic.pdf")
plot_roc_curve(y_true, y_prob, "Basic Logistic Regression")

# ---------- STANDARDIZED LR
print("\n--- Starting Logistic Regression With Scaling Pipeline ---")
assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
categorical_cols_indexed = [f"{col}_index" for col in categorical_cols]
assembler_categorical = VectorAssembler(inputCols=categorical_cols_indexed, outputCol="categorical_features")
scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
assembler_final = VectorAssembler(inputCols=["scaled_numeric_features", "categorical_features"], outputCol="features")
train, test = df.randomSplit([0.8, 0.2])
pipeline_std = Pipeline(stages=[
    assembler_numeric,
    scaler,
    assembler_categorical,
    assembler_final,
    LogisticRegression(labelCol="HadHeartAttack_Index", featuresCol="features")
])
cv_std = CrossValidator(estimator=pipeline_std, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
start_std = time.time()
model_std = cv_std.fit(train)
elapsed_std = time.time() - start_std
train_pred_std = model_std.transform(train)
test_pred_std = model_std.transform(test)
train_acc_std = evaluator.evaluate(train_pred_std)
test_acc_std = evaluator.evaluate(test_pred_std)
auc_train_std = compute_auc(train_pred_std)
auc_test_std = compute_auc(test_pred_std)
print(f"STANDARDIZED - Train Acc: {train_acc_std}, Test Acc: {test_acc_std}, Train AUC: {auc_train_std}, Test AUC: {auc_test_std}, Time: {elapsed_std}")
save_results("standardized_logistic_regression.txt",  model_std.bestModel.stages[-1], train_acc_std, test_acc_std, elapsed_std,feature_columns, auc_train_std, auc_test_std)
paired_std = test_pred_std.select("HadHeartAttack_Index", "prediction", "probability").rdd \
    .filter(lambda row: row["prediction"] is not None) \
    .map(lambda row: (row["HadHeartAttack_Index"], row["prediction"], row["probability"][1])) \
    .collect()
y_true_std, y_pred_std, y_prob_std = zip(*paired_std)
y_true_std = np.array(y_true_std)
y_pred_std = np.array(y_pred_std)
y_prob_std = np.array(y_prob_std)
plot_confusion_matrix(y_true_std, y_pred_std, "confusion_matrix_standardized.pdf")
plot_roc_curve(y_true_std, y_prob_std, "Standardized Logistic Regression")

# ---------- PCA-based LR
print("\n--- Starting PCA-based Logistic Regression Pipeline ---")

# Compute Scree Plot and Save Variance Table
assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
heart_data_scree = assembler_numeric.transform(df)

assembler_categorical = VectorAssembler(inputCols=[f"{col}_index" for col in categorical_cols], outputCol="categorical_features")
heart_data_scree = assembler_categorical.transform(heart_data_scree)

assembler_combined = VectorAssembler(inputCols=["numeric_features"] + [f"{col}_index" for col in categorical_cols],
                                     outputCol="combined_features")
heart_data_scree = assembler_combined.transform(heart_data_scree)

num_components = len(numeric_cols) + len(categorical_cols)
pca_scree = PCA(k=num_components, inputCol="combined_features", outputCol="pca_features")
pca_model_scree = pca_scree.fit(heart_data_scree)

explained_variance = pca_model_scree.explainedVariance.toArray()

# Save components + variance table
pc_variance_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(num_components)],
    "Explained Variance Ratio": explained_variance
})
pc_variance_df.to_csv("pca_explained_variance.csv", index=False)
print("Saved PCA explained variance table to pca_explained_variance.csv")

# Plot scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_components + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot - Explained Variance by Principal Component")
plt.xticks(range(1, num_components + 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_scree_plot.pdf")
plt.close()
print("Saved PCA scree plot to pca_scree_plot.pdf")

for k in [5, 10, 15]:
    print(f"\nStandardized Logistic Regression with PCA k={k}")
    
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    feature_columns = [f"PC{i+1}" for i in range(k)]
    
    assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
    assembler_categorical = VectorAssembler(inputCols=[f"{col}_index" for col in categorical_cols], outputCol="categorical_features")
    assembler_all_for_pca = VectorAssembler(inputCols=["scaled_numeric_features", "categorical_features"], outputCol="combined_features")
    pca = PCA(k=k, inputCol="combined_features", outputCol="features")
    lr = LogisticRegression(labelCol="HadHeartAttack_Index", featuresCol="features")
    
    pipeline_pca = Pipeline(stages=[
        assembler_numeric,
        scaler,
        assembler_categorical,
        assembler_all_for_pca,
        pca,
        lr
    ])
    
    cv_pca = CrossValidator(estimator=pipeline_pca, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    start_pca = time.time()
    model_pca = cv_pca.fit(train)
    elapsed_pca = time.time() - start_pca
    
    train_pred_pca = model_pca.transform(train)
    test_pred_pca = model_pca.transform(test)
    train_acc_pca = evaluator.evaluate(train_pred_pca)
    test_acc_pca = evaluator.evaluate(test_pred_pca)
    auc_train_pca = compute_auc(train_pred_pca)
    auc_test_pca = compute_auc(test_pred_pca)
    
    print(f"PCA-{k} based - Train Acc: {train_acc_pca}, Test Acc: {test_acc_pca}, Train AUC: {auc_train_pca}, Test AUC: {auc_test_pca}, Time: {elapsed_pca}")
    
    save_results(f"logistic_regression_model_pca_{k}.txt", model_pca.bestModel.stages[-1], train_acc_pca, test_acc_pca, elapsed_pca, feature_columns, auc_train_pca, auc_test_pca)
    paired_pca = test_pred_pca.select("HadHeartAttack_Index", "prediction", "probability").rdd \
    .filter(lambda row: row["prediction"] is not None) \
    .map(lambda row: (row["HadHeartAttack_Index"], row["prediction"], row["probability"][1])) \
    .collect()
    y_true_pca, y_pred_pca, y_prob_pca = zip(*paired_pca)
    y_true_pca = np.array(y_true_pca)
    y_pred_pca = np.array(y_pred_pca)
    y_prob_pca = np.array(y_prob_pca)
       
    plot_confusion_matrix(y_true_pca, y_pred_pca, f"confusion_matrix_pca_{k}.pdf")
    plot_roc_curve(y_true_pca, y_prob_pca, f"PCA k={k}")

# Final ROC comparison
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_comparison.pdf")
plt.close()

spark.stop()
