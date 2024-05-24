import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_process_data(filepath, label_column):
    # 读取数据
    data = pd.read_csv(filepath, delimiter=',')

    # 打印列名以检查是否包含 'risk_score' 和 'anomaly'
    print("Columns in the dataset:", data.columns)

    # 打印数据集的前几行
    print("First few rows of the dataset:\n", data.head())

    # 将字符串类型的风险评分转化为数值型
    if 'risk_score' in data.columns:
        risk_mapping = {'low_risk': 0, 'moderate_risk': 1, 'high_risk': 2}
        data['risk_score'] = data['risk_score'].map(risk_mapping)

    # 将字符串类型的异常转化为数值型
    if 'anomaly' in data.columns:
        anomaly_mapping = {'low_risk': 0, 'moderate_risk': 1, 'high_risk': 2}
        data['anomaly'] = data['anomaly'].map(anomaly_mapping)

    # 编码分类特征
    categorical_features = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # 特征和标签
    X = data.drop(['timestamp', 'sending_address', 'receiving_address', label_column], axis=1)
    y = data[label_column]

    # 确保所有数据都是数值型的
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.fillna(0).astype(int)

    # 检查 y 的类别分布
    print(f"Value counts of {label_column} before split:", y.value_counts())

    # 数据分割：70%用于训练，30%用于测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 检查 y_train 的类别分布
    print(f"Value counts of {label_column} after split:", y_train.value_counts())

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    filepath = '../dataset/metaverse_transactions_dataset.csv'

    # 使用 anomaly 作为标签
    print("Using anomaly as label:")
    X_train, X_test, y_train, y_test = load_and_process_data(filepath, 'anomaly')
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
