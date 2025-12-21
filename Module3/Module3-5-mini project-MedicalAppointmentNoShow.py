#Module 3-5-mini-project-Editor Medical Appointment
import pandas as pd
# dataset = pd.read_csv('Medical_Appointment.csv')
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

api = KaggleApi()
api.authenticate()

api.dataset_download_files('joniarroba/noshowappointments',
                           path='data', unzip=True)

# make sure nRowsRead is defined before using it
nRowsRead = 1000
df = pd.read_csv('data/KaggleV2-May-2016.csv', nrows=nRowsRead)
# Does the dataset include any missing values? 
missing_values = df.isnull().sum()
#print("Missing values in each column:\n", missing_values)
# Feature Extraction
# Extract the following features:
# Gender
# Age
# Scholarship
# Hipertension
# Diabetes
# Alcoholism
# Handcap
# SMS_received
selected_features = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
y = df['No-show'].map({'No': 0, 'Yes': 1}).values
X = df[selected_features]
X.describe()
# Preprocessing: Perform any needed pre-processing of the chosen features including:
# Scaling
# Encoding
# Dealing with Nan values
# Identify categorical and numeric columns
# Save categorical columns in `cat_cols` (explicit and robust)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
print(f"Numeric cols ({len(num_cols)}): {num_cols}")

# Create an encoded copy for scoring: label-encode categorical features so mutual information operates on original features
X_encoded = X.copy()

for c in cat_cols:
    le = LabelEncoder()
    X_encoded[c] = le.fit_transform(X_encoded[c].astype(str))

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_encoded[num_cols] = imputer.fit_transform(X_encoded[num_cols]) 

# Split your data as follows:
# 80% training set
# 10% validation set
# 10% test set
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=0)  # 0.1111 x 0.9 ≈ 0.1

# Training Tree-based Classifiers
# Use a decision-tree classifier model to train your data.
# Choose the best criterion for the decision tree algorithm by trying different values and validating performance on the validation set.
tree_model=DecisionTreeClassifier(
    criterion='gini',
    random_state=0
)
tree_model.fit(X_train, y_train)
y_val_pred_tree = tree_model.predict(X_val)
val_score_tree = accuracy_score(y_val, y_val_pred_tree)
print('Decision Tree Validation Accuracy: ' + str(val_score_tree))


# Random Forest
# Repeat step 6.
# Increase/decrease the number of estimators in random forest and comment on the difference of the classification metrics.
# Choose k (top features to keep)
k = min(5, X_encoded.shape[1])
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
