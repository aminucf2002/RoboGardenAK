from sklearn.datasets import fetch_openml
import pandas as pd

# load dataset
data = fetch_openml(name='credit-g', version=1)
# Read the dataset into a Pandas DataFrame!
df = pd.DataFrame(data.data, columns=data.feature_names)

#perform a feature selection to identify and retain only the most relevant features for classification.
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# add target series
y = pd.Series(data.target, name='target')
print("\nTarget distribution:", y.value_counts())
      
# Identify categorical and numeric columns
# Save categorical columns in `cat_cols` (explicit and robust)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols]

print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
print(f"Numeric cols ({len(num_cols)}): {num_cols}")

# Create an encoded copy for scoring: label-encode categorical features so mutual information operates on original features
X_encoded = df.copy()
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_encoded[c] = le.fit_transform(X_encoded[c].astype(str))
    label_encoders[c] = le


# Choose k (top features to keep)
k = min(10, X_encoded.shape[1])
selector = SelectKBest(mutual_info_classif, k=k)
selector.fit(X_encoded, y)

scores = selector.scores_
feature_scores = pd.Series(scores, index=X_encoded.columns).sort_values(ascending=False)

print("\nTop features by mutual information score:")
print(feature_scores.head(k))

# Pick top-k features and create a reduced DataFrame
top_features = feature_scores.head(k).index.tolist()
print("\nSelected top-{} features: {}".format(k, top_features))

# Reduced dataframe containing only selected features (original columns)
df_reduced = df[top_features].copy()
print("\nReduced DataFrame shape:", df_reduced.shape)

# Quick model-based check: train RandomForest on encoded top features to get importances
rf = RandomForestClassifier(random_state=0, n_estimators=100)
rf.fit(X_encoded[top_features], y)
importances = pd.Series(rf.feature_importances_, index=top_features).sort_values(ascending=False)
print("\nRandomForest importances on selected features:")
print(importances)

# Save selected columns for downstream modelling
selected_columns = top_features

df_reduced.head() 

# Perform any needed pre-processing on the chosen features, includes:
# Scaling
# Encoding
# Dealing with NaN values
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

cat_cols_reduced = df_reduced.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols_reduced = [c for c in df_reduced.columns if c not in cat_cols_reduced]

df_reduced[num_cols_reduced] = imputer.fit_transform(df_reduced[num_cols_reduced])
df_reduced[num_cols_reduced] = scaler.fit_transform(df_reduced[num_cols_reduced])
for c in cat_cols_reduced:
    if c in df_reduced.columns:
        le = label_encoders[c]
        df_reduced[c] = le.transform(df_reduced[c].astype(str))
        
y_encode=LabelEncoder().fit_transform(y)
# Split your data as follows:
# 80% training set
# 10% validation set
# 10% test set
X_train_val, X_test, y_train_val, y_test = train_test_split(df_reduced, y_encode, test_size=0.1, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=0)  # 0.1111 x 0.9 ≈ 0.1
# Training Classifiers
# Use the KNN-classifier model to train your data.
# Choose the best k for the KNN algorithm by trying different values and validating performance on the validation set.
# Print the accuracy score of your final classifier.
# print the confusion matrix.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_val_pred_knn = model_knn.predict(X_val)
val_score_knn = accuracy_score(y_val, y_val_pred_knn)
print('KNN Validation Accuracy: ' + str(val_score_knn))