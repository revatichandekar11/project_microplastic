from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
from PIL import Image
import cv2

# Load your locally saved YOLOv8 model
model = YOLO('models/best.pt')  # Path to your downloaded model

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load the image with OpenCV for processing
        img = cv2.imread(filepath)

        # Run YOLOv8 model for inference
        results = model.predict(img)  # Perform inference
        
        # Prepare class information (example)
        class_info = {
            0: "Low-Density Polyethylene (LDPE) with dust:Known for its flexibility and used in food packaging.",
            1: "Polyethylene (PE):Commonly found in bags and films.",
            2: "Polyethylene (PE) with dust:Commonly found in bags and films.",
            3: "Polyhydroxyalkanoate (PHA):A biodegradable alternative gaining traction.",
            4: "Polyhydroxyalkanoate (PHA) with dust:A biodegradable alternative gaining traction.",
            5: "Polystyrene (PS):Used in packaging and disposable items.",
            6:"Polystyrene (PS) with dust:Used in packaging and disposable items.",
            # Add more class descriptions as needed
        }

        detected_classes = []  # List to hold detected class IDs

        # Draw results on the image
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Get class labels
            
            for i in range(len(boxes)):
                box = boxes[i]
                confidence = confidences[i]
                class_id = int(classes[i])
                
                detected_classes.append(class_id)  # Add class ID to detected list

                # Draw bounding box
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                label = f'Class: {class_id}, Conf: {confidence:.2f}'
                cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the result image
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(result_image_path, img)  # Save the processed image with bounding boxes

        # Render the result and pass detected classes
        return render_template('result.html', original=filepath, result=result_image_path, 
                               class_ids=detected_classes, class_info=class_info)

if __name__ == '__main__':
    app.run(debug=True)
