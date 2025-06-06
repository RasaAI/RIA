# Let’s clean our bank dataset!
import pandas as pd

df = pd.read_csv(r'/Users/surya/Downloads/back end/backend/BankLiquidityRiskDetectionNEW.csv')

df.drop('Unnamed: 0',axis = 1,inplace=True)
df.head(10)


# Print column names for debugging
print("Dataset columns:", df.columns.tolist())

# Check initial size
print("Original shape:", df.shape)

# Remove duplicates
print("Removing duplicates...")
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# Check for missing values
print("Checking for missing values...")
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Verify no missing values
print("Missing values after filling:\n", df.isnull().sum()[df.isnull().sum() > 0])


# Drop irrelevant columns
print("Dropping unnecessary columns...")
drop_cols = ['REPORTINGDATE', 'XX_MLA_CLASS2', 'MLA_CLASS2']
df = df.drop(columns=(col for col in drop_cols if col in df.columns))

# Save cleaned data
print("Saving cleaned data...")
df.to_csv('cleaned_bank_data.csv', index=False)
print("Cleaned shape:", df.shape)


# Let’s prepare the data!
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load cleaned data
print("Loading cleaned_bank_data.csv...")
df = pd.read_csv('cleaned_bank_data.csv')

# Print columns for debugging
print("Cleaned dataset columns:", df.columns.tolist())


# Split into features (X) and target (y)
print("Splitting features and target...")
X = df.drop(columns=['EWL_LIQUIDITY RATING'])
y = df['EWL_LIQUIDITY RATING']


# Define categorical column
cat_cols = ['INSTITUTIONCODE'] if 'INSTITUTIONCODE' in X.columns else []
# Define numerical columns (exclude categorical and ensure they exist)
num_cols = [col for col in X.columns if col not in cat_cols]

# Verify columns
missing_cols = [col for col in num_cols + cat_cols if col not in X.columns]
if missing_cols:
    print("Warning: These columns are missing:", missing_cols)
    num_cols = [col for col in num_cols if col in X.columns]
    cat_cols = [col for col in cat_cols if col in X.columns]

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)


# Create preprocessing pipeline
print("Setting up preprocessing...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])


# Split data: 70% train, 15% validation, 15% test
print("Splitting into train, validation, test...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42  # 0.1765 = 15/(100-15)
)

# Check shapes
print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)


# Save splits
print("Saving data splits...")
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Let’s pick the best features!
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# Load training data
print("Loading training data...")
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Print columns for debugging
print("Training columns:", X_train.columns.tolist())

# Create pipeline
print("Building feature selection pipeline...")
feature_selector = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


# Fit on training data
print("Finding important features...")
feature_selector.fit(X_train, y_train)

# Get feature importances
rf = feature_selector.named_steps['classifier']
importances = rf.feature_importances_

# Get feature names after preprocessing
num_features = num_cols
cat_features = feature_selector.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols) if cat_cols else []
all_features = np.concatenate([num_features, cat_features])


# Create importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Select top 20 features, ensuring they exist in X_train
top_features = [f for f in feature_importance_df.head(20)['Feature'].tolist() if f in X_train.columns or f in cat_features]
print("Top 20 features (filtered):\n", top_features)


# Filter datasets
print("Filtering train, val, test sets...")
X_train_selected = X_train[[col for col in top_features if col in X_train.columns]]
X_val = pd.read_csv('X_val.csv')
X_val_selected = X_val[[col for col in top_features if col in X_val.columns]]
X_test = pd.read_csv('X_test.csv')
X_test_selected = X_test[[col for col in top_features if col in X_test.columns]]

# Save selected features
print("Saving selected features...")
X_train_selected.to_csv('X_train_selected.csv', index=False)
X_val_selected.to_csv('X_val_selected.csv', index=False)
X_test_selected.to_csv('X_test_selected.csv', index=False)
print("Selected train shape:", X_train_selected.shape)


print("Saving selected features to selected_features.csv...")
feature_importance_df.head(20)[['Feature', 'Importance']].to_csv('selected_features.csv', index=False)
print("Selected features saved to selected_features.csv")


# import matplotlib.pyplot as plt
# import seaborn as sns
# # Visualize feature importances
# print("Creating feature importance visualization...")
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), color='#1f77b4')
# plt.title('Top 20 Feature Importances for Liquidity Risk Prediction')
# plt.xlabel('Importance Score')
# plt.ylabel('Feature')
# plt.tight_layout()



# Let’s try Voting Ensemble!
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
Xtrain = pd.read_csv('X_train_selected.csv')
ytrain = pd.read_csv('y_train.csv').values.ravel()
Xval = pd.read_csv('X_val_selected.csv')
yval = pd.read_csv('y_val.csv').values.ravel()
Xtest = pd.read_csv('X_test_selected.csv')
ytest = pd.read_csv('y_test.csv').values.ravel()

# Create model
model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42))
    ],
    voting='soft'
)

model.fit(Xtrain,ytrain)

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
