import cv2
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, Response, request, jsonify
from torchvision import models
import torch.nn as nn
import os

app = Flask(_name_)

# Load the YOLOv5 model (object detection)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
target_classes = ['cell phone', 'laptop']

# Load the classification model (for phones: brand & model)
class_model = models.resnet18(pretrained=False)  # Assuming you're using ResNet18
num_ftrs = class_model.fc.in_features
class_model.fc = nn.Linear(num_ftrs, 2)  # Assuming two classes: 'Apple iphone', 'vivo IQ Z6 lite'
class_model.load_state_dict(torch.load('laptop_classifier.pth'))
class_model.eval()

# Define the classes for the classification model
class_names = ['Apple iphone', 'vivo IQ Z6 lite']

# Define transformations for classification model input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define the video capture
cap = cv2.VideoCapture(0)

# To store the latest captured frame
latest_frame = None
detection_details = {}

def count_objects(results, class_names):
    counts = {class_name: 0 for class_name in class_names}
    for det in results.xyxy[0]:  # Results.xyxy contains [x1, y1, x2, y2, confidence, class]
        class_id = int(det[5])
        class_name = yolo_model.names[class_id]
        if class_name in class_names:
            counts[class_name] += 1
    return counts

def generate_frames():
    global latest_frame, detection_details
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection (stage 1)
        results = yolo_model(frame)
        results.render()  # Draw bounding boxes
        counts = count_objects(results, target_classes)

        # Update detection details for displaying in GUI
        detection_details.update(counts)

        # Display object counts on the frame
        cv2.putText(frame, f"Cell Phones: {counts['cell phone']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Laptops: {counts['laptop']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # If a cell phone is detected, run classification model (stage 2)
        if counts['cell phone'] > 0:
            # Convert the frame to be compatible with the ResNet model
            image_tensor = transform(frame).unsqueeze(0)

            # Run the classification model
            with torch.no_grad():
                outputs = class_model(image_tensor)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds[0]]

            # Update detection details with the classification result
            detection_details['classification'] = predicted_class

            # Draw the predicted class (brand/model) on the frame
            cv2.putText(frame, predicted_class, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store the latest frame for image capture
        latest_frame = frame.copy()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    if latest_frame is not None:
        # Save the latest frame as an image
        filename = os.path.join('static', 'captured_image.jpg')
        cv2.imwrite(filename, latest_frame)
        return jsonify({'message': 'Image captured successfully!', 'image_url': filename, 'details': detection_details})
    else:
        return jsonify({'message': 'No frame available to capture!'})

if _name_ == '_main_':
    app.run(debug=True)