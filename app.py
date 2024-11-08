import os
import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash
import uuid



# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flashing messages
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
print("Upload folder:", app.config['UPLOAD_FOLDER'])


# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None

    # Perform age and gender analysis
    try:
        analysis = DeepFace.analyze(image, actions=['age', 'gender'], enforce_detection=False)
        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']
        return age, gender
    except Exception as e:
        print("Error during analysis:", e)
        return None

def analyze_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frame_count = 0
    ages = []
    genders = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        frame_count += 1
        if frame_count % 100 == 0:  # Analyze every 100th frame
            try:
                analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
                ages.append(analysis[0]['age'])
                genders.append(analysis[0]['dominant_gender'])
            except Exception as e:
                print("Error during analysis:", e)

    cap.release()

    if ages:
        median_age = np.median(ages)
        most_common_gender = Counter(genders).most_common(1)[0][0]
        return median_age, most_common_gender
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            # Save the file with a unique name
            # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)


            # Determine if the file is an image or video
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                result = analyze_image(filepath)
            elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                result = analyze_video(filepath)
            else:
                flash('Unsupported file format. Please upload an image or video file.')
                return redirect(request.url)

            if result:
                age, gender = result
                return render_template('index.html', age=age, gender=gender, filename=filename)
            else:
                flash('Error processing file. Please try again.')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
