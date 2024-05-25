import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelEncoder,StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# 绘制宏平均ROC曲线的函数
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
    # 加载数据
    data = pd.read_csv('metaverse_transactions_dataset.csv')
    # Convert timestamp to datetime and extract year and month
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month

    # Encode categorical features
    categorical_features = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly']
    label_encoders = {feature: LabelEncoder() for feature in categorical_features}

    for feature in categorical_features:
        data[feature] = label_encoders[feature].fit_transform(data[feature])

    # Select features and target
    features = ['hour_of_day', 'amount', 'transaction_type', 'location_region', 'ip_prefix',
                'login_frequency', 'session_duration', 'purchase_pattern', 'age_group','risk_score'
                , 'year', 'month']
    target = 'anomaly'

    X = data[features]
    y = data[target]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 初始化SMOTE对象

    #smote = SMOTE( k_neighbors=10, random_state=42)

    # 进行过采样
    #X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型
    xgb_model = XGBClassifier(n_estimators=30, max_depth=2, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    #xgb_model = XGBClassifier(n_estimators=100, max_depth=10, random_state=42)
    '''
    xgb_model = XGBClassifier(
        n_estimators=10,  # 更少的树的数量100
        max_depth=2,  # 更小的树深度10
        subsample=0.6,  # 更低的行采样比例
        colsample_bytree=0.6,  # 更低的特征采样比例
        learning_rate=0.1,  # 保持适当的学习速率
        use_label_encoder=False,  # 禁用旧的标签编码方式
        eval_metric='mlogloss',  # 使用多类对数损失作为评估指标
        random_state=42  # 设定随机种子以确保结果的一致性
    )
'''
    xgb_model.fit(X_train, y_train)

    # 定义输出路径
    output_path = 'roc_curve_XGBoost.png'

    # 绘制宏平均ROC曲线并保存图像
    plot_macro_averaged_roc_curve(xgb_model, X_test, y_test, output_path)