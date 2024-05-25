from src.data_processing import load_and_process_data
from src.train_model_o import train_logistic_regression, train_knn
from src.evaluate_model import evaluate_model, print_results

if __name__ == "__main__":
    # 数据路径
    filepath = 'dataset/metaverse_transactions_dataset.csv'

    # 加载和处理数据
    X_train, X_test, y_train, y_test = load_and_process_data(filepath, 'anomaly')

    # 确保数据集包含多个类别
    if len(y_train.unique()) < 2:
        print("Error: The label 'anomaly' contains less than two classes. Please check the dataset.")
    else:
        # 训练Logistic Regression模型
        log_reg_model = train_logistic_regression(X_train, y_train)
        # 评估Logistic Regression模型
        log_reg_report1, log_reg_accuracy1 = evaluate_model(log_reg_model, X_train, y_train)
        log_reg_report, log_reg_accuracy = evaluate_model(log_reg_model, X_test, y_test)
        # 打印Logistic Regression结果
        print("\nTrain:Logistic Regression Results")

        print_results(log_reg_report1, log_reg_accuracy1)
        print("\nTest:Logistic Regression Results")
        print_results(log_reg_report, log_reg_accuracy)

        # 训练KNN模型
        knn_model = train_knn(X_train, y_train)
        # 评估KNN模型
        knn_report1, knn_accuracy1 = evaluate_model(knn_model, X_train, y_train)
        knn_report, knn_accuracy = evaluate_model(knn_model, X_test, y_test)
        # 打印KNN结果
        print("\nTrain:KNN Results")
        print_results(knn_report1, knn_accuracy1)
        print("\nTest:KNN Results:")
        print_results(knn_report, knn_accuracy)
