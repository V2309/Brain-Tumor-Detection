
├── app/
│   ├── static/                 # CSS, JS, and images for the Flask web app
│   ├── templates/              # HTML templates for the Flask app
│   └── app.py                  # Main Flask application

│__test_images
|
├── model/
│   └── brain_tumor_model.pth   # Pretrained PyTorch model
│
├── data/
│   ├── train/                  # Training MRI images
│   ├── test/                   # Testing MRI images
│
├── src/
│   ├── dataset.py              # Script to load and preprocess the dataset
│   ├── model.py                # CNN model architecture using PyTorch
│   └── train.py                # Script to train the model
│
├── README.md                   # Project documentation
└── requirements.txt            # List of required Python packages