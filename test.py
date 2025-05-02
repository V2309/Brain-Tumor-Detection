import os

BASE_DIR = "data"  # thư mục chứa train và test
classes = ['notumor', 'glioma', 'meningioma', 'pituitary']

for split in ['train', 'test']:
    print(f"\n📂 {split.upper()} SET")
    for cls in classes:
        folder_path = os.path.join(BASE_DIR, split, cls)
        num_files = len(os.listdir(folder_path))
        print(f"{cls:12}: có {num_files} ảnh")

