import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

def plot_macro_averaged_roc_curve(model, X_test, y_test, output_path):
    # 将标签二值化
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # 获取预测概率
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        raise ValueError("The model does not support predict_proba method.")

    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)

    # 确保曲线从(0,0)开始到(1,1)结束
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    # 绘制ROC曲线
    plt.figure()
    plt.plot(all_fpr, mean_tpr, color='orange', label=f'Macro-Averaged ROC curve (AUC = {macro_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # 保存图像
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the dataset
    file_path = '../metaverse_transactions_dataset.csv'
    dataset = pd.read_csv(file_path)

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

    # # Train the SVM model
    # svm_model = SVC(kernel='linear', random_state=42, probability=True)
    # svm_model.fit(X_train, y_train)
    #
    # # Define output path for the ROC curve image
    # output_path = 'macro_averaged_roc_curve_SVM.png'
    #
    # # Plot macro-averaged ROC curve and save the image
    # plot_macro_averaged_roc_curve(svm_model, X_test, y_test, output_path)

    # 去除最重要特征 risk_score
    features_reduced = ['hour_of_day', 'amount', 'transaction_type', 'location_region', 'ip_prefix',
                        'login_frequency', 'session_duration', 'purchase_pattern', 'age_group',
                        'year', 'month']
    X_reduced = dataset[features_reduced]

    # Split the data into training and testing sets
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.2,
                                                                                        random_state=42)

    # Standardize numerical features
    X_train_reduced = scaler.fit_transform(X_train_reduced)
    X_test_reduced = scaler.transform(X_test_reduced)

    # Train the SVM model
    svm_model_reduced = SVC(kernel='linear', random_state=42)
    svm_model_reduced.fit(X_train_reduced, y_train_reduced)

    y_train_pred_reduced = svm_model_reduced.predict(X_train_reduced)
    y_test_pred_reduced = svm_model_reduced.predict(X_test_reduced)

    train_accuracy_reduced = accuracy_score(y_train_reduced, y_train_pred_reduced)
    test_accuracy_reduced = accuracy_score(y_test_reduced, y_test_pred_reduced)

    train_report_reduced = classification_report(y_train_reduced, y_train_pred_reduced,
                                                 target_names=label_encoders['anomaly'].classes_)
    test_report_reduced = classification_report(y_test_reduced, y_test_pred_reduced,
                                                target_names=label_encoders['anomaly'].classes_)

    # print("SVM_MODEL")
    print("SVM_model without risk_score:")
    print("======================================================")
    print(train_report_reduced)
    print(f"Accuracy of TRAIN data: {train_accuracy_reduced:}")
    print("======================================================")
    print(test_report_reduced)
    print(f"Accuracy of TEST data: {test_accuracy_reduced}")
    print("======================================================")

    # # Output classification report
    # y_pred = svm_model.predict(X_test)
    # report = classification_report(y_test, y_pred, target_names=label_encoders['anomaly'].classes_)
    # print("Classification Report:")
    # print(report)

    # Train the SVM model
    svm_reduced_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_reduced_model.fit(X_train_reduced, y_train_reduced)

    # Define output path for the ROC curve image
    output_path = 'macro_averaged_roc_curve_SVM_reduced.png'

    # Plot macro-averaged ROC curve and save the image
    plot_macro_averaged_roc_curve(svm_reduced_model, X_test_reduced, y_test_reduced, output_path)
