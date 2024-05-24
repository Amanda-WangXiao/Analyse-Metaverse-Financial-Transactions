from src.data_processing import load_and_process_data
from src.train_model_o import train_knn

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data('./dataset/metaverse_transactions_dataset.csv')
    train_knn(X_train, y_train)
