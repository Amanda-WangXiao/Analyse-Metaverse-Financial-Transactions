import pandas as pd

# Load the dataset
file_path = '../metaverse_transactions_dataset.csv'
dataset = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Convert timestamp to datetime and extract year and month
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
dataset['year'] = dataset['timestamp'].dt.year
dataset['month'] = dataset['timestamp'].dt.month

# Encode categorical features
categorical_features = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly']
label_encoders = {feature: LabelEncoder() for feature in categorical_features}

for feature in categorical_features:
    dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])

# Select features and target
features = ['hour_of_day', 'amount', 'transaction_type', 'location_region', 'ip_prefix',
            'login_frequency', 'session_duration', 'purchase_pattern', 'age_group',
            'risk_score', 'year', 'month']
target = 'anomaly'

X = dataset[features]
y = dataset[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_report = classification_report(y_train, y_train_pred, target_names=label_encoders['anomaly'].classes_)
test_report = classification_report(y_test, y_test_pred, target_names=label_encoders['anomaly'].classes_)

print("SVM_MODEL")
print("======================================================")
print(train_report)
print(f"Accuracy of TRAIN data: {train_accuracy:}")
print("======================================================")
print(test_report)
print(f"Accuracy of TEST data: {test_accuracy:}")
print("======================================================")
