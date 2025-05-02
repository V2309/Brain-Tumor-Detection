# # src/train.py
# import torch
# from dataset import get_data_loaders
# import os
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib

# def train_knn(batch_size=64):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Sử dụng thiết bị: {device}")
    
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     train_dir = os.path.join(base_dir, 'data', 'train')
#     test_dir = os.path.join(base_dir, 'data', 'test')
    
#     train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)
#     print(f"Training samples: {len(train_loader.dataset)}")
#     print(f"Testing samples: {len(test_loader.dataset)}")
    
#     # Extract training features
#     train_features = []
#     train_labels = []
#     print("Extracting training features...")
#     for i, (features, labels) in enumerate(train_loader):
#         print(f"Batch {i+1}/{len(train_loader)} processed")
#         train_features.append(features)
#         train_labels.append(labels)
#     train_features = np.concatenate(train_features, axis=0)
#     train_labels = np.concatenate(train_labels, axis=0)
#     print(f"Train features shape: {train_features.shape}")
    
#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=50)
#     train_features_pca = pca.fit_transform(train_features)
#     print(f"Train features shape after PCA: {train_features_pca.shape}")
    
#     # Grid search for optimal KNN parameters
#     param_grid = {
#         'n_neighbors': [3, 5, 7, 9],
#         'weights': ['uniform', 'distance']
#     }
#     knn = KNeighborsClassifier()
#     grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(train_features_pca, train_labels)
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
    
#     # Train final model with best parameters
#     best_knn = grid_search.best_estimator_
    
#     # Extract test features
#     test_features = []
#     test_labels = []
#     print("Extracting testing features...")
#     for i, (features, labels) in enumerate(test_loader):
#         print(f"Batch {i+1}/{len(test_loader)} processed")
#         test_features.append(features)
#         test_labels.append(labels)
#     test_features = np.concatenate(test_features, axis=0)
#     test_labels = np.concatenate(test_labels, axis=0)
#     test_features_pca = pca.transform(test_features)
#     print(f"Test features shape after PCA: {test_features_pca.shape}")
    
#     # Evaluate
#     test_accuracy = best_knn.score(test_features_pca, test_labels)
#     print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
#     print("Classification Report:")
#     print(classification_report(test_labels, best_knn.predict(test_features_pca), 
#                                 target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
#     print("Confusion Matrix:")
#     print(confusion_matrix(test_labels, best_knn.predict(test_features_pca)))
    
#     # Save the PCA and KNN models
#     model_dir = os.path.join(base_dir, 'model')
#     os.makedirs(model_dir, exist_ok=True)
#     joblib.dump(best_knn, os.path.join(model_dir, 'knn_model.pkl'))
#     joblib.dump(pca, os.path.join(model_dir, 'pca_model.pkl'))
#     print(f"Models saved to {model_dir}")

# if __name__ == "__main__":
#     train_knn(batch_size=64)



# src/train.py
# import torch
# from dataset import get_data_loaders
# import os
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC  # Import SVM classifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# import time  # Import thư viện time

# def train_knn_svm(batch_size=64):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Sử dụng thiết bị: {device}")
    
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     train_dir = os.path.join(base_dir, 'data', 'train')
#     test_dir = os.path.join(base_dir, 'data', 'test')
    
#     train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)
#     print(f"Số mẫu huấn luyện: {len(train_loader.dataset)}")
#     print(f"Số mẫu kiểm tra: {len(test_loader.dataset)}")
    
#     # Extract training features
#     train_features = []
#     train_labels = []
#     print("Extracting training features...")
#     for i, (features, labels) in enumerate(train_loader):
        
#         print(f"Batch {i+1}/{len(train_loader)} processed")
#         train_features.append(features)
#         train_labels.append(labels)
     
#     train_features = np.concatenate(train_features, axis=0)
#     train_labels = np.concatenate(train_labels, axis=0)
#     print(f"Kích thước đặc trưng huấn luyện: {train_features.shape}")
    
#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=50)
#     train_features_pca = pca.fit_transform(train_features)
#     print(f"Kích thước đặc trưng sau PCA: {train_features_pca.shape}")
    
#     # --- KNN Training ---
#     print("\n--- Training KNN ---")
#     knn_param_grid = {
#         'n_neighbors': [3, 5, 7, 9],
#         'weights': ['uniform', 'distance']
#     }
#     knn = KNeighborsClassifier()
#     knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     knn_grid_search.fit(train_features_pca, train_labels)
#     print(f"KNN tham số tốt nhất: {knn_grid_search.best_params_}")
#     print(f"KNN Độ chính xác cross-validation tốt nhất: {knn_grid_search.best_score_ * 100:.2f}%")
#     best_knn = knn_grid_search.best_estimator_
    
#     # --- SVM Training ---
#     print("\n--- Training SVM ---")
#     svm_param_grid = {
#         'C': [0.1, 1, 10],  # Regularization parameter
#         'kernel': ['linear', 'rbf'],  # Linear and RBF kernels
#         'gamma': ['scale', 'auto']  # Kernel coefficient
#     }
#     svm = SVC(probability=True)  # Enable probability for predict_proba
#     svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     svm_grid_search.fit(train_features_pca, train_labels)
#     print(f"SVM tham số tốt nhất: {svm_grid_search.best_params_}")
#     print(f"SVM Độ chính xác cross-validation tốt nhất: {svm_grid_search.best_score_ * 100:.2f}%")
#     best_svm = svm_grid_search.best_estimator_
    
#     # Extract test features
#     test_features = []
#     test_labels = []
#     print("\nExtracting testing features...")
#     for i, (features, labels) in enumerate(test_loader):
       
#         print(f"Batch {i+1}/{len(test_loader)} processed")
#         test_features.append(features)
#         test_labels.append(labels)
#     test_features = np.concatenate(test_features, axis=0)
#     test_labels = np.concatenate(test_labels, axis=0)
#     test_features_pca = pca.transform(test_features)
#     print(f"Kích thước đặc trưng kiểm tra sau PCA: {test_features_pca.shape}")
    
#     # --- Evaluate KNN ---
#     print("\n--- KNN Evaluation ---")
#     knn_test_accuracy = best_knn.score(test_features_pca, test_labels)
#     print(f"KNN Độ chính xác trên tập kiểm tra: {knn_test_accuracy * 100:.2f}%")
#     print("KNN Báo cáo phân loại::")
#     print(classification_report(test_labels, best_knn.predict(test_features_pca), 
#                                 target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
#     print("KNN Ma trận nhầm lẫn:")
#     print(confusion_matrix(test_labels, best_knn.predict(test_features_pca)))
    
#     # --- Evaluate SVM ---
#     print("\n--- SVM Evaluation ---")
#     svm_test_accuracy = best_svm.score(test_features_pca, test_labels)
#     print(f"SVM Độ chính xác trên tập kiểm tra: {svm_test_accuracy * 100:.2f}%")
#     print("SVM Báo cáo phân loại:")
#     print(classification_report(test_labels, best_svm.predict(test_features_pca), 
#                                 target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
#     print("SVM Ma trận nhầm lẫn:")
#     print(confusion_matrix(test_labels, best_svm.predict(test_features_pca)))
    
#     # Save both models
#     model_dir = os.path.join(base_dir, 'model')
#     os.makedirs(model_dir, exist_ok=True)
#     joblib.dump(best_knn, os.path.join(model_dir, 'knn_model.pkl'))
#     joblib.dump(best_svm, os.path.join(model_dir, 'svm_model.pkl'))
#     joblib.dump(pca, os.path.join(model_dir, 'pca_model.pkl'))
#     print(f"Mô hình đã được lưu: {model_dir}")

# if __name__ == "__main__":
#     train_knn_svm(batch_size=64)

#-----------------------------------------
# #train.pypy
# import torch
# from dataset import get_data_loaders
# import os
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# def plot_confusion_matrix(cm, classes, title, save_path):
#     """
#     Vẽ và lưu ma trận nhầm lẫn.
#     """
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.title(title)
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.savefig(save_path)
#     plt.close()

# def plot_accuracy_comparison(datasets, knn_accuracies, svm_accuracies, save_path):
#     """
#     Vẽ biểu đồ so sánh độ chính xác của KNN và SVM trên các tập dữ liệu.
#     """
#     plt.figure(figsize=(10, 6))
#     x = np.arange(len(datasets))
#     width = 0.35

#     plt.bar(x - width/2, knn_accuracies, width, label='KNN', color='skyblue')
#     plt.bar(x + width/2, svm_accuracies, width, label='SVM', color='lightcoral')
#     plt.xlabel('Dataset')
#     plt.ylabel('Test Accuracy (%)')
#     plt.title('KNN vs SVM Accuracy Comparison')
#     plt.xticks(x, datasets)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def train_knn_svm(data_dirs, batch_size=64):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Sử dụng thiết bị: {device}")
    
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     model_dir = os.path.join(base_dir, 'model')
#     os.makedirs(model_dir, exist_ok=True)
    
#     results = {}
#     knn_accuracies = []
#     svm_accuracies = []
#     dataset_names = []
    
#     for data_dir_name in data_dirs:
#         print(f"\n=== Xử lý tập dữ liệu: {data_dir_name} ===")
#         train_dir = os.path.join(base_dir, data_dir_name, 'train')
#         test_dir = os.path.join(base_dir, data_dir_name, 'test')
        
#         train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)
#         print(f"Số mẫu huấn luyện: {len(train_loader.dataset)}")
#         print(f"Số mẫu kiểm tra: {len(test_loader.dataset)}")
        
#         # Extract training features
#         train_features = []
#         train_labels = []
#         print("Extracting training features...")
#         for i, (features, labels) in enumerate(train_loader):
            
#             print(f"Batch {i+1}/{len(train_loader)} processed")
#             train_features.append(features)
#             train_labels.append(labels)
        
#         train_features = np.concatenate(train_features, axis=0)
#         train_labels = np.concatenate(train_labels, axis=0)
#         print(f"Kích thước đặc trưng huấn luyện: {train_features.shape}")
        
#         # Apply PCA for dimensionality reduction
#         pca = PCA(n_components=50)
#         train_features_pca = pca.fit_transform(train_features)
#         print(f"Kích thước đặc trưng sau PCA: {train_features_pca.shape}")
        
#         # --- KNN Training ---
#         print("\n--- Training KNN ---")
#         knn_param_grid = {
#             'n_neighbors': [3, 5, 7, 9],
#             'weights': ['uniform', 'distance']
#         }
#         knn = KNeighborsClassifier()
#         knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#         knn_grid_search.fit(train_features_pca, train_labels)
#         print(f"KNN tham số tốt nhất: {knn_grid_search.best_params_}")
#         print(f"KNN Độ chính xác cross-validation tốt nhất: {knn_grid_search.best_score_ * 100:.2f}%")
#         best_knn = knn_grid_search.best_estimator_
        
#         # --- SVM Training ---
#         print("\n--- Training SVM ---")
#         svm_param_grid = {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         }
#         svm = SVC(probability=True)
#         svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#         svm_grid_search.fit(train_features_pca, train_labels)
#         print(f"SVM tham số tốt nhất: {svm_grid_search.best_params_}")
#         print(f"SVM Độ chính xác cross-validation tốt nhất: {svm_grid_search.best_score_ * 100:.2f}%")
#         best_svm = svm_grid_search.best_estimator_
        
#         # Extract test features
#         test_features = []
#         test_labels = []
#         print("\nExtracting testing features...")
#         for i, (features, labels) in enumerate(test_loader):
#             print(f"Batch {i+1}/{len(test_loader)} processed")
#             test_features.append(features)
#             test_labels.append(labels)
#         test_features = np.concatenate(test_features, axis=0)
#         test_labels = np.concatenate(test_labels, axis=0)
#         test_features_pca = pca.transform(test_features)
#         print(f"Kích thước đặc trưng kiểm tra sau PCA: {test_features_pca.shape}")
        
#         # --- Evaluate KNN ---
#         print("\n--- KNN Evaluation ---")
#         knn_test_accuracy = best_knn.score(test_features_pca, test_labels)
#         print(f"KNN Độ chính xác trên tập kiểm tra: {knn_test_accuracy * 100:.2f}%")
#         knn_report = classification_report(test_labels, best_knn.predict(test_features_pca), 
#                                           target_names=['glioma', 'meningioma', 'notumor', 'pituitary'])
#         print("KNN Báo cáo phân loại:")
#         print(knn_report)
#         knn_cm = confusion_matrix(test_labels, best_knn.predict(test_features_pca))
#         print("KNN Ma trận nhầm lẫn:")
#         print(knn_cm)
        
#         # --- Evaluate SVM ---
#         print("\n--- SVM Evaluation ---")
#         svm_test_accuracy = best_svm.score(test_features_pca, test_labels)
#         print(f"SVM Độ chính xác trên tập kiểm tra: {svm_test_accuracy * 100:.2f}%")
#         svm_report = classification_report(test_labels, best_svm.predict(test_features_pca), 
#                                           target_names=['glioma', 'meningioma', 'notumor', 'pituitary'])
#         print("SVM Báo cáo phân loại:")
#         print(svm_report)
#         svm_cm = confusion_matrix(test_labels, best_svm.predict(test_features_pca))
#         print("SVM Ma trận nhầm lẫn:")
#         print(svm_cm)
        
#         # Save models
#         dataset_model_dir = os.path.join(model_dir, data_dir_name)
#         os.makedirs(dataset_model_dir, exist_ok=True)
#         joblib.dump(best_knn, os.path.join(dataset_model_dir, 'knn_model.pkl'))
#         joblib.dump(best_svm, os.path.join(dataset_model_dir, 'svm_model.pkl'))
#         joblib.dump(pca, os.path.join(dataset_model_dir, 'pca_model.pkl'))
#         print(f"Mô hình đã được lưu: {dataset_model_dir}")
        
#         # Visualize confusion matrices
#         plot_confusion_matrix(knn_cm, ['glioma', 'meningioma', 'notumor', 'pituitary'],
#                             f"KNN Confusion Matrix ({data_dir_name})",
#                             os.path.join(dataset_model_dir, 'knn_confusion_matrix.png'))
#         plot_confusion_matrix(svm_cm, ['glioma', 'meningioma', 'notumor', 'pituitary'],
#                             f"SVM Confusion Matrix ({data_dir_name})",
#                             os.path.join(dataset_model_dir, 'svm_confusion_matrix.png'))
        
#         # Store results
#         results[data_dir_name] = {
#             'knn_accuracy': knn_test_accuracy * 100,
#             'svm_accuracy': svm_test_accuracy * 100,
#             'knn_report': knn_report,
#             'svm_report': svm_report
#         }
#         knn_accuracies.append(knn_test_accuracy * 100)
#         svm_accuracies.append(svm_test_accuracy * 100)
#         dataset_names.append(data_dir_name)
    
#     # Plot accuracy comparison
#     plot_accuracy_comparison(dataset_names, knn_accuracies, svm_accuracies,
#                            os.path.join(model_dir, 'accuracy_comparison.png'))
    
#     # Print comparison summary
#     print("\n=== So sánh kết quả ===")
#     for dataset in results:
#         print(f"\nTập dữ liệu: {dataset}")
#         print(f"KNN Độ chính xác: {results[dataset]['knn_accuracy']:.2f}%")
#         print(f"SVM Độ chính xác: {results[dataset]['svm_accuracy']:.2f}%")
#         print("KNN Báo cáo phân loại:")
#         print(results[dataset]['knn_report'])
#         print("SVM Báo cáo phân loại:")
#         print(results[dataset]['svm_report'])

# if __name__ == "__main__":
#     data_dirs = ['data', 'dataset']  # Danh sách các thư mục dữ liệu
#     train_knn_svm(data_dirs, batch_size=64)












import torch
from dataset import get_data_loaders
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_confusion_matrix(cm, classes, title, save_path):
    """
    Vẽ và lưu ma trận nhầm lẫn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_comparison(datasets, knn_accuracies, svm_accuracies, save_path):
    """
    Vẽ biểu đồ so sánh độ chính xác của KNN và SVM trên các tập dữ liệu.
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35

    plt.bar(x - width/2, knn_accuracies, width, label='KNN', color='skyblue')
    plt.bar(x + width/2, svm_accuracies, width, label='SVM', color='lightcoral')
    plt.xlabel('Dataset')
    plt.ylabel('Test Accuracy (%)')
    plt.title('KNN vs SVM Accuracy Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_and_loss(datasets, knn_train_acc, knn_val_acc, svm_train_acc, svm_val_acc, 
                           knn_train_loss, knn_val_loss, svm_train_loss, svm_val_loss, save_path):
    """
    Vẽ biểu đồ so sánh Training và Validation Accuracy, Training và Validation Loss cho KNN và SVM.
    """
    plt.figure(figsize=(12, 10))

    # Plot Accuracy
    plt.subplot(2, 1, 1)
    x = np.arange(len(datasets))
    width = 0.2

    plt.bar(x - width*1.5, knn_train_acc, width, label='KNN Train Accuracy', color='skyblue')
    plt.bar(x - width/2, knn_val_acc, width, label='KNN Validation Accuracy', color='lightblue')
    plt.bar(x + width/2, svm_train_acc, width, label='SVM Train Accuracy', color='lightcoral')
    plt.bar(x + width*1.5, svm_val_acc, width, label='SVM Validation Accuracy', color='coral')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(2, 1, 2)
    plt.bar(x - width*1.5, knn_train_loss, width, label='KNN Train Loss', color='skyblue')
    plt.bar(x - width/2, knn_val_loss, width, label='KNN Validation Loss', color='lightblue')
    plt.bar(x + width/2, svm_train_loss, width, label='SVM Train Loss', color='lightcoral')
    plt.bar(x + width*1.5, svm_val_loss, width, label='SVM Validation Loss', color='coral')
    
    plt.xlabel('Dataset')
    plt.ylabel('Loss (1 - Accuracy)')
    plt.title('Training and Validation Loss Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_knn_svm(data_dirs, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    results = {}
    knn_accuracies = []
    svm_accuracies = []
    knn_train_acc = []
    knn_val_acc = []
    svm_train_acc = []
    svm_val_acc = []
    knn_train_loss = []
    knn_val_loss = []
    svm_train_loss = []
    svm_val_loss = []
    dataset_names = []
    
    for data_dir_name in data_dirs:
        print(f"\n=== Xử lý tập dữ liệu: {data_dir_name} ===")
        train_dir = os.path.join(base_dir, data_dir_name, 'train')
        test_dir = os.path.join(base_dir, data_dir_name, 'test')
        
        train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)
        print(f"Số mẫu huấn luyện: {len(train_loader.dataset)}")
        print(f"Số mẫu kiểm tra: {len(test_loader.dataset)}")
        
        # Extract training features
        train_features = []
        train_labels = []
        print("Extracting training features...")
        for i, (features, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed")
            train_features.append(features.cpu().numpy())  # Chuyển sang CPU và numpy
            train_labels.append(labels.cpu().numpy())
            import gc
            gc.collect()  # Giải phóng bộ nhớ
        
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        print(f"Kích thước đặc trưng huấn luyện: {train_features.shape}")
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=50)
        train_features_pca = pca.fit_transform(train_features)
        print(f"Kích thước đặc trưng sau PCA: {train_features_pca.shape}")
        
        # --- KNN Training ---
        print("\n--- Training KNN ---")
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
        knn = KNeighborsClassifier()
        knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        knn_grid_search.fit(train_features_pca, train_labels)
        print(f"KNN tham số tốt nhất: {knn_grid_search.best_params_}")
        print(f"KNN Độ chính xác cross-validation tốt nhất: {knn_grid_search.best_score_ * 100:.2f}%")
        best_knn = knn_grid_search.best_estimator_
        
        # Calculate training accuracy and loss for KNN
        knn_train_accuracy = best_knn.score(train_features_pca, train_labels)
        knn_val_accuracy = knn_grid_search.best_score_
        knn_train_loss_value = 1 - knn_train_accuracy  # Proxy loss
        knn_val_loss_value = 1 - knn_val_accuracy      # Proxy loss
        
        # --- SVM Training ---
        print("\n--- Training SVM ---")
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        svm = SVC(probability=True)
        svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid_search.fit(train_features_pca, train_labels)
        print(f"SVM tham số tốt nhất: {svm_grid_search.best_params_}")
        print(f"SVM Độ chính xác cross-validation tốt nhất: {svm_grid_search.best_score_ * 100:.2f}%")
        best_svm = svm_grid_search.best_estimator_
        
        # Calculate training accuracy and loss for SVM
        svm_train_accuracy = best_svm.score(train_features_pca, train_labels)
        svm_val_accuracy = svm_grid_search.best_score_
        svm_train_loss_value = 1 - svm_train_accuracy  # Proxy loss
        svm_val_loss_value = 1 - svm_val_accuracy      # Proxy loss
        
        # Extract test features
        test_features = []
        test_labels = []
        print("\nExtracting testing features...")
        for i, (features, labels) in enumerate(test_loader):
            print(f"Batch {i+1}/{len(test_loader)} processed")
            test_features.append(features.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            gc.collect()
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_features_pca = pca.transform(test_features)
        print(f"Kích thước đặc trưng kiểm tra sau PCA: {test_features_pca.shape}")
        
        # --- Evaluate KNN ---
        print("\n--- KNN Evaluation ---")
        knn_test_accuracy = best_knn.score(test_features_pca, test_labels)
        print(f"KNN Độ chính xác trên tập kiểm tra: {knn_test_accuracy * 100:.2f}%")
        knn_report = classification_report(test_labels, best_knn.predict(test_features_pca), 
                                          target_names=['glioma', 'meningioma', 'notumor', 'pituitary'])
        print("KNN Báo cáo phân loại:")
        print(knn_report)
        knn_cm = confusion_matrix(test_labels, best_knn.predict(test_features_pca))
        print("KNN Ma trận nhầm lẫn:")
        print(knn_cm)
        
        # --- Evaluate SVM ---
        print("\n--- SVM Evaluation ---")
        svm_test_accuracy = best_svm.score(test_features_pca, test_labels)
        print(f"SVM Độ chính xác trên tập kiểm tra: {svm_test_accuracy * 100:.2f}%")
        svm_report = classification_report(test_labels, best_svm.predict(test_features_pca), 
                                          target_names=['glioma', 'meningioma', 'notumor', 'pituitary'])
        print("SVM Báo cáo phân loại:")
        print(svm_report)
        svm_cm = confusion_matrix(test_labels, best_svm.predict(test_features_pca))
        print("SVM Ma trận nhầm lẫn:")
        print(svm_cm)
        
        # Save models
        dataset_model_dir = os.path.join(model_dir, data_dir_name)
        os.makedirs(dataset_model_dir, exist_ok=True)
        joblib.dump(best_knn, os.path.join(dataset_model_dir, 'knn_model.pkl'))
        joblib.dump(best_svm, os.path.join(dataset_model_dir, 'svm_model.pkl'))
        joblib.dump(pca, os.path.join(dataset_model_dir, 'pca_model.pkl'))
        print(f"Mô hình đã được lưu: {dataset_model_dir}")
        
        # Visualize confusion matrices
        plot_confusion_matrix(knn_cm, ['glioma', 'meningioma', 'notumor', 'pituitary'],
                            f"KNN Confusion Matrix ({data_dir_name})",
                            os.path.join(dataset_model_dir, 'knn_confusion_matrix.png'))
        plot_confusion_matrix(svm_cm, ['glioma', 'meningioma', 'notumor', 'pituitary'],
                            f"SVM Confusion Matrix ({data_dir_name})",
                            os.path.join(dataset_model_dir, 'svm_confusion_matrix.png'))
        
        # Store results
        results[data_dir_name] = {
            'knn_accuracy': knn_test_accuracy * 100,
            'svm_accuracy': svm_test_accuracy * 100,
            'knn_report': knn_report,
            'svm_report': svm_report,
            'knn_train_accuracy': knn_train_accuracy * 100,
            'knn_val_accuracy': knn_val_accuracy * 100,
            'svm_train_accuracy': svm_train_accuracy * 100,
            'svm_val_accuracy': svm_val_accuracy * 100
        }
        knn_accuracies.append(knn_test_accuracy * 100)
        svm_accuracies.append(svm_test_accuracy * 100)
        knn_train_acc.append(knn_train_accuracy * 100)
        knn_val_acc.append(knn_val_accuracy * 100)
        svm_train_acc.append(svm_train_accuracy * 100)
        svm_val_acc.append(svm_val_accuracy * 100)
        knn_train_loss.append(knn_train_loss_value * 100)
        knn_val_loss.append(knn_val_loss_value * 100)
        svm_train_loss.append(svm_train_loss_value * 100)
        svm_val_loss.append(svm_val_loss_value * 100)
        dataset_names.append(data_dir_name)
    
    # Plot accuracy comparison
    plot_accuracy_comparison(dataset_names, knn_accuracies, svm_accuracies,
                           os.path.join(model_dir, 'accuracy_comparison.png'))
    
    # Plot training and validation accuracy/loss
    plot_accuracy_and_loss(dataset_names, knn_train_acc, knn_val_acc, svm_train_acc, svm_val_acc,
                           knn_train_loss, knn_val_loss, svm_train_loss, svm_val_loss,
                           os.path.join(model_dir, 'train_val_accuracy_loss.png'))
    
    # Print comparison summary
    print("\n=== So sánh kết quả ===")
    for dataset in results:
        print(f"\nTập dữ liệu: {dataset}")
        print(f"KNN Độ chính xác (Test): {results[dataset]['knn_accuracy']:.2f}%")
        print(f"SVM Độ chính xác (Test): {results[dataset]['svm_accuracy']:.2f}%")
        print(f"KNN Độ chính xác (Train): {results[dataset]['knn_train_accuracy']:.2f}%")
        print(f"KNN Độ chính xác (Validation): {results[dataset]['knn_val_accuracy']:.2f}%")
        print(f"SVM Độ chính xác (Train): {results[dataset]['svm_train_accuracy']:.2f}%")
        print(f"SVM Độ chính xác (Validation): {results[dataset]['svm_val_accuracy']:.2f}%")
        print("KNN Báo cáo phân loại:")
        print(results[dataset]['knn_report'])
        print("SVM Báo cáo phân loại:")
        print(results[dataset]['svm_report'])

if __name__ == "__main__":
    data_dirs = ['data', 'dataset']  # Danh sách các thư mục dữ liệu
    train_knn_svm(data_dirs, batch_size=64)