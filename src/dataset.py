# # src/dataset.py
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import resnet18, ResNet18_Weights
# from PIL import Image
# import os
# import glob
# import numpy as np
# import torch.nn as nn

# # Lớp BrainTumorDataset để xử lý dữ liệu hình ảnh u não
# class BrainTumorDataset(Dataset):
#     def __init__(self, data_dir, transform=None, feature_extractor=None):
#         """
#         Hàm khởi tạo cho lớp BrainTumorDataset.
#         - data_dir: Đường dẫn đến thư mục chứa dữ liệu.
#         - transform: Các phép biến đổi áp dụng lên ảnh.
#         - feature_extractor: Bộ trích xuất đặc trưng (ví dụ: ResNet18).
#         """
#         self.data_dir = data_dir
#         self.transform = transform
#         self.feature_extractor = feature_extractor
        
#         # Danh sách các lớp (4 loại u não)
#         self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Ánh xạ lớp sang chỉ số
        
#         self.image_paths = []  # Danh sách đường dẫn ảnh
#         self.labels = []       # Danh sách nhãn tương ứng
        
#         print(f"Loading data from: {data_dir}")  # Thông báo đang tải dữ liệu
#         for class_name in self.classes:
#             class_dir = os.path.join(data_dir, class_name)  # Đường dẫn đến thư mục của từng lớp
#             print(f"Checking: {class_dir}")
            
#             # Tìm tất cả các ảnh trong thư mục (hỗ trợ nhiều định dạng)
#             images = (glob.glob(os.path.join(class_dir, '*.png')) + 
#                      glob.glob(os.path.join(class_dir, '*.jpg')) + 
#                      glob.glob(os.path.join(class_dir, '*.JPG')) + 
#                      glob.glob(os.path.join(class_dir, '*.jpeg')))
            
#             print(f"Found {len(images)} images in {class_name}")  # Số lượng ảnh tìm thấy
#             print(f"Assigning label {self.class_to_idx[class_name]} to {len(images)} images in {class_name}")
#             self.image_paths.extend(images)  # Thêm đường dẫn ảnh vào danh sách
#             self.labels.extend([self.class_to_idx[class_name]] * len(images))  # Thêm nhãn tương ứng
        
#         # Kiểm tra nếu không tìm thấy ảnh nào
#         if not self.image_paths:
#             raise ValueError(f"No images found in {data_dir}! Check directory structure and file extensions.")
        
#         print(f"Total samples: {len(self.image_paths)}, Labels distribution: {np.bincount(self.labels)}")

#     def __len__(self):
#         """
#         Trả về số lượng mẫu trong tập dữ liệu.
#         """
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """
#         Lấy một mẫu dữ liệu dựa trên chỉ số (idx).
#         - idx: Chỉ số của mẫu cần lấy.
#         """
#         img_path = self.image_paths[idx]  # Đường dẫn ảnh
#         label = self.labels[idx]          # Nhãn tương ứng
        
#         img = Image.open(img_path)  # Mở ảnh
#         if img.mode != 'RGB':       # Chuyển ảnh sang chế độ RGB nếu cần
#             img = img.convert('RGB')
        
#         if self.transform:          # Áp dụng các phép biến đổi (nếu có)
#             img = self.transform(img)
        
#         if self.feature_extractor:  # Nếu có bộ trích xuất đặc trưng
#             with torch.no_grad():   # Tắt gradient để tăng tốc
#                 features = self.feature_extractor(img.unsqueeze(0)).squeeze().cpu().numpy()  # Trích xuất đặc trưng
#             return features, label  # Trả về đặc trưng và nhãn
#         return img, label           # Trả về ảnh và nhãn nếu không có bộ trích xuất đặc trưng

# # Hàm để tạo DataLoader cho tập huấn luyện và kiểm tra
# def get_data_loaders(train_dir, test_dir, batch_size=32):
#     """
#     Tạo DataLoader cho tập huấn luyện và kiểm tra.
#     - train_dir: Đường dẫn đến thư mục chứa dữ liệu huấn luyện.
#     - test_dir: Đường dẫn đến thư mục chứa dữ liệu kiểm tra.
#     - batch_size: Kích thước batch.
#     """
#     # Các phép biến đổi áp dụng lên ảnh
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh về 224x224
#         transforms.ToTensor(),         # Chuyển ảnh sang tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
#     ])
    
#     # Tải mô hình ResNet18 để trích xuất đặc trưng
#     resnet = resnet18(weights=ResNet18_Weights.DEFAULT)  # Sử dụng trọng số đã được huấn luyện trước
#     resnet.fc = nn.Identity()  # Loại bỏ lớp fully connected cuối cùng
#     resnet.eval()              # Đặt mô hình ở chế độ đánh giá (evaluation mode)
#     resnet.to("cpu")           # Chuyển mô hình sang CPU (có thể đổi sang GPU nếu cần)
    
#     # Tạo tập dữ liệu huấn luyện và kiểm tra
#     train_dataset = BrainTumorDataset(train_dir, transform=transform, feature_extractor=resnet)
#     test_dataset = BrainTumorDataset(test_dir, transform=transform, feature_extractor=resnet)
    
#     # Tạo DataLoader cho tập huấn luyện và kiểm tra
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     return train_loader, test_loader

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import glob
import numpy as np
import torch.nn as nn

# Lớp BrainTumorDataset để xử lý dữ liệu hình ảnh u não
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None, feature_extractor=None):
        """
        Hàm khởi tạo cho lớp BrainTumorDataset.
        - data_dir: Đường dẫn đến thư mục chứa dữ liệu (ví dụ: 'data/train', 'dataset/test').
        - transform: Các phép biến đổi áp dụng lên ảnh.
        - feature_extractor: Bộ trích xuất đặc trưng (ví dụ: ResNet18).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.feature_extractor = feature_extractor
        
        # Danh sách các lớp (4 loại u não)
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Ánh xạ lớp sang chỉ số
        
        self.image_paths = []  # Danh sách đường dẫn ảnh
        self.labels = []       # Danh sách nhãn tương ứng
        
        print(f"Loading data from: {data_dir}")  # Thông báo đang tải dữ liệu
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)  # Đường dẫn đến thư mục của từng lớp
            print(f"Checking: {class_dir}")
            
            # Tìm tất cả các ảnh trong thư mục (hỗ trợ nhiều định dạng)
            images = (glob.glob(os.path.join(class_dir, '*.png')) + 
                     glob.glob(os.path.join(class_dir, '*.jpg')) + 
                     glob.glob(os.path.join(class_dir, '*.JPG')) + 
                     glob.glob(os.path.join(class_dir, '*.jpeg')))
            
            print(f"Found {len(images)} images in {class_name}")  # Số lượng ảnh tìm thấy
            print(f"Assigning label {self.class_to_idx[class_name]} to {len(images)} images in {class_name}")
            self.image_paths.extend(images)  # Thêm đường dẫn ảnh vào danh sách
            self.labels.extend([self.class_to_idx[class_name]] * len(images))  # Thêm nhãn tương ứng
        
        # Kiểm tra nếu không tìm thấy ảnh nào
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}! Check directory structure and file extensions.")
        
        print(f"Total samples: {len(self.image_paths)}, Labels distribution: {np.bincount(self.labels)}")

    def __len__(self):  
        """
        Trả về số lượng mẫu trong tập dữ liệu.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu dựa trên chỉ số (idx).
        - idx: Chỉ số của mẫu cần lấy.
        """
        img_path = self.image_paths[idx]  # Đường dẫn ảnh
        label = self.labels[idx]          # Nhãn tương ứng
        
        img = Image.open(img_path)  # Mở ảnh
        if img.mode != 'RGB':       # Chuyển ảnh sang chế độ RGB nếu cần
            img = img.convert('RGB')
        
        if self.transform:          # Áp dụng các phép biến đổi (nếu có)
            img = self.transform(img)
        
        if self.feature_extractor:  # Nếu có bộ trích xuất đặc trưng
            with torch.no_grad():   # Tắt gradient để tăng tốc
                features = self.feature_extractor(img.unsqueeze(0)).squeeze().cpu().numpy()  # Trích xuất đặc trưng
            return features, label  # Trả về đặc trưng và nhãn
        return img, label           # Trả về ảnh và nhãn nếu không có bộ trích xuất đặc trưng

# Hàm để tạo DataLoader cho tập huấn luyện và kiểm tra
def get_data_loaders(train_dir, test_dir, batch_size=64):
    """
    Tạo DataLoader cho tập huấn luyện và kiểm tra.
    - train_dir: Đường dẫn đến thư mục chứa dữ liệu huấn luyện.
    - test_dir: Đường dẫn đến thư mục chứa dữ liệu kiểm tra.
    - batch_size: Kích thước batch.
    """
    # Các phép biến đổi áp dụng lên ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh về 224x224
        transforms.ToTensor(),         # Chuyển ảnh sang tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
    ])
    
    # Tải mô hình ResNet18 để trích xuất đặc trưng
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)  # Sử dụng trọng số đã được huấn luyện trước
    resnet.fc = nn.Identity()  # Loại bỏ lớp fully connected cuối cùng
    resnet.eval()              # Đặt mô hình ở chế độ đánh giá (evaluation mode)
    resnet.to("cpu")           # Chuyển mô hình sang CPU (có thể đổi sang GPU nếu cần)
    
    # Tạo tập dữ liệu huấn luyện và kiểm tra
    train_dataset = BrainTumorDataset(train_dir, transform=transform, feature_extractor=resnet)
    test_dataset = BrainTumorDataset(test_dir, transform=transform, feature_extractor=resnet)
    
    # Tạo DataLoader cho tập huấn luyện và kiểm tra
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader