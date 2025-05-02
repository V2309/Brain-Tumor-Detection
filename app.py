# # app/app.py
# import flask
# from io import BytesIO
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from torchvision.transforms import Compose, ToTensor, Resize, Normalize
# from torchvision.models import resnet18, ResNet18_Weights
# import os
# import numpy as np
# import joblib
# import torch.nn as nn

# upload = os.path.join('static', 'photos')
# app = flask.Flask(__name__, template_folder='templates')
# app.secret_key = "secret key"
# app.config['upload'] = upload
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Updated to 4 labels
# LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Sử dụng thiết bị: {device}")

# # Load feature extractor (ResNet18)
# resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
# resnet.fc = nn.Identity()
# resnet.eval()
# resnet.to(device)

# # Load PCA and KNN models
# pca_model = joblib.load('./model/pca_model.pkl')
# knn_model = joblib.load('./model/knn_model.pkl')


# def preprocess_image(image_bytes):
#     transform = Compose([
#         Resize((224, 224)),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img = Image.open(BytesIO(image_bytes))
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return transform(img).unsqueeze(0), img  # Return tensor and original PIL image

# def get_prediction(image_bytes):
#     tensor, original_img = preprocess_image(image_bytes=image_bytes)
#     with torch.no_grad():
#         features = resnet(tensor.to(device)).squeeze().cpu().numpy()
#     features_pca = pca_model.transform([features])[0]
#     probabilities = knn_model.predict_proba([features_pca])[0]
#     class_id = knn_model.predict([features_pca])[0]
#     class_name = LABELS[class_id]
#     confidence = probabilities[class_id]  # Confidence score for predicted class
#     print(f"Probabilities: {probabilities} (Glioma, Meningioma, No Tumor, Pituitary)")
#     print(f"Predicted class ID: {class_id}")

#     # If not "No Tumor" (class_id != 2), add a bounding box with label and confidence
#     if class_id != 2:  # Not "No Tumor"
#         draw = ImageDraw.Draw(original_img)
#         # Define bounding box (centered on image)
#         width, height = original_img.size
#         box_size = min(width, height) // 2  # Box size is half the smallest dimension
#         left = (width - box_size) // 2
#         top = (height - box_size) // 2
#         right = left + box_size
#         bottom = top + box_size
#         draw.rectangle(
#             [left, top, right, bottom],
#             outline="red", width=5  # Red bounding box with thickness 5
#         )

#         # Add label and confidence score above the box
#         label_text = f"{class_name} {confidence:.2f}"
#         try:
#             # Use a default font if available
#             font = ImageFont.truetype("arial.ttf", 20)
#         except:
#             # Fallback to default PIL font if truetype font is unavailable
#             font = ImageFont.load_default()
#         text_bbox = draw.textbbox((0, 0), label_text, font=font)
#         text_width = text_bbox[2] - text_bbox[0]
#         text_height = text_bbox[3] - text_bbox[1]
#         text_x = left + (box_size - text_width) // 2
#         text_y = top - text_height - 5  # Place above the box
#         # Draw a background rectangle for the text
#         draw.rectangle(
#             [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
#             fill="red"
#         )
#         draw.text((text_x, text_y), label_text, fill="white", font=font)
    
#     return str(class_id), class_name, original_img  # Return class_id, class_name, and modified image

# @app.route('/', methods=['GET'])
# def main():
#     return flask.render_template('DiseaseDet.html')

# @app.route("/uimg", methods=['GET', 'POST'])
# def uimg():
#     if flask.request.method == 'GET':
#         return flask.render_template('uimg.html')
#     if flask.request.method == 'POST':
#         file = flask.request.files['file']
#         if file and allowed_file(file.filename):
#             filename = file.filename
#             filepath = os.path.join(app.config['upload'], filename)
#             os.makedirs(app.config['upload'], exist_ok=True)
#             file.save(filepath)
            
#             img_bytes = file.stream.read() if file.stream.tell() == 0 else open(filepath, 'rb').read()
#             class_id, class_name, modified_img = get_prediction(img_bytes)
            
#             # Save the modified image with a new filename
#             modified_filename = f"result_{filename}"
#             modified_filepath = os.path.join(app.config['upload'], modified_filename)
#             modified_img.save(modified_filepath)
            
#             # URLs for original and modified images
#             file_url = flask.url_for('static', filename=f'photos/{filename}')
#             modified_file_url = flask.url_for('static', filename=f'photos/{modified_filename}')
            
#             return flask.render_template(
#                 'pred.html', 
#                 result=class_name, 
#                 file_url=file_url, 
#                 modified_file_url=modified_file_url
#             )
#         return "File not allowed", 400

# @app.errorhandler(500)
# def server_error(error):
#     return flask.render_template('error.html'), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# app/app.py
import flask
from io import BytesIO
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet18, ResNet18_Weights
import os
import numpy as np
import joblib
import torch.nn as nn

upload = os.path.join('static', 'photos')
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['upload'] = upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Updated to 4 labels
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Load feature extractor (ResNet18)
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()
resnet.to(device)

# Load PCA, KNN, and SVM models
pca_model = joblib.load('./model/pca_model.pkl')
knn_model = joblib.load('./model/knn_model.pkl')
svm_model = joblib.load('./model/svm_model.pkl')  # Added SVM model loading

def preprocess_image(image_bytes):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return transform(img).unsqueeze(0), img  # Return tensor and original PIL image

def get_prediction(image_bytes):
    tensor, original_img = preprocess_image(image_bytes=image_bytes)
    with torch.no_grad():
        features = resnet(tensor.to(device)).squeeze().cpu().numpy()
    features_pca = pca_model.transform([features])[0]

    # --- KNN Prediction ---
    knn_probabilities = knn_model.predict_proba([features_pca])[0]
    knn_class_id = knn_model.predict([features_pca])[0]
    knn_class_name = LABELS[knn_class_id]
    knn_confidence = knn_probabilities[knn_class_id]  # Confidence score for predicted class
    print(f"KNN Probabilities: {knn_probabilities} (Glioma, Meningioma, No Tumor, Pituitary)")
    print(f"KNN Predicted class ID: {knn_class_id}")

    # Create a copy of the original image for KNN
    knn_img = original_img.copy()
    # If not "No Tumor" (class_id != 2), add a bounding box with label and confidence
    if knn_class_id != 2:  # Not "No Tumor"
        draw = ImageDraw.Draw(knn_img)
        # Define bounding box (centered on image)
        width, height = knn_img.size
        box_size = min(width, height) // 2  # Box size is half the smallest dimension
        left = (width - box_size) // 2
        top = (height - box_size) // 2
        right = left + box_size
        bottom = top + box_size
        draw.rectangle(
            [left, top, right, bottom],
            outline="red", width=5  # Red bounding box with thickness 5
        )

        # Add label and confidence score above the box
        label_text = f"KNN: {knn_class_name} {knn_confidence:.2f}"
        try:
            # Use a default font if available
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default PIL font if truetype font is unavailable
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = left + (box_size - text_width) // 2
        text_y = top - text_height - 5  # Place above the box
        # Draw a background rectangle for the text
        draw.rectangle(
            [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
            fill="red"
        )
        draw.text((text_x, text_y), label_text, fill="white", font=font)

    # --- SVM Prediction ---
    svm_probabilities = svm_model.predict_proba([features_pca])[0]
    svm_class_id = svm_model.predict([features_pca])[0]
    svm_class_name = LABELS[svm_class_id]
    svm_confidence = svm_probabilities[svm_class_id]  # Confidence score for predicted class
    print(f"SVM Probabilities: {svm_probabilities} (Glioma, Meningioma, No Tumor, Pituitary)")
    print(f"SVM Predicted class ID: {svm_class_id}")

    # Create a copy of the original image for SVM
    svm_img = original_img.copy()
    # If not "No Tumor" (class_id != 2), add a bounding box with label and confidence
    if svm_class_id != 2:  # Not "No Tumor"
        draw = ImageDraw.Draw(svm_img)
        # Define bounding box (centered on image)
        width, height = svm_img.size
        box_size = min(width, height) // 2  # Box size is half the smallest dimension
        left = (width - box_size) // 2
        top = (height - box_size) // 2
        right = left + box_size
        bottom = top + box_size
        draw.rectangle(
            [left, top, right, bottom],
            outline="red", width=5  # Red bounding box with thickness 5
        )

        # Add label and confidence score above the box
        label_text = f"SVM: {svm_class_name} {svm_confidence:.2f}"
        try:
            # Use a default font if available
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default PIL font if truetype font is unavailable
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = left + (box_size - text_width) // 2
        text_y = top - text_height - 5  # Place above the box
        # Draw a background rectangle for the text
        draw.rectangle(
            [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
            fill="red"
        )
        draw.text((text_x, text_y), label_text, fill="white", font=font)

    return (knn_class_id, knn_class_name, knn_img, knn_probabilities), (svm_class_id, svm_class_name, svm_img, svm_probabilities)

@app.route('/', methods=['GET'])
def main():
    return flask.render_template('DiseaseDet.html')

@app.route("/uimg", methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['upload'], filename)
            os.makedirs(app.config['upload'], exist_ok=True)
            file.save(filepath)
            
            img_bytes = file.stream.read() if file.stream.tell() == 0 else open(filepath, 'rb').read()
            (knn_class_id, knn_class_name, knn_img, knn_prob), (svm_class_id, svm_class_name, svm_img, svm_prob) = get_prediction(img_bytes)
            
            # Save the modified images with new filenames
            knn_modified_filename = f"knn_result_{filename}"
            knn_modified_filepath = os.path.join(app.config['upload'], knn_modified_filename)
            knn_img.save(knn_modified_filepath)

            svm_modified_filename = f"svm_result_{filename}"
            svm_modified_filepath = os.path.join(app.config['upload'], svm_modified_filename)
            svm_img.save(svm_modified_filepath)
            
            # URLs for original and modified images
            file_url = flask.url_for('static', filename=f'photos/{filename}')
            knn_modified_file_url = flask.url_for('static', filename=f'photos/{knn_modified_filename}')
            svm_modified_file_url = flask.url_for('static', filename=f'photos/{svm_modified_filename}')
            
            return flask.render_template(
                'pred.html', 
                knn_result=knn_class_name,
                svm_result=svm_class_name,
                knn_prob=knn_prob,
                svm_prob=svm_prob,
                file_url=file_url,
                knn_modified_file_url=knn_modified_file_url,
                svm_modified_file_url=svm_modified_file_url
            )
        return "File not allowed", 400

@app.errorhandler(500)
def server_error(error):
    return flask.render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)