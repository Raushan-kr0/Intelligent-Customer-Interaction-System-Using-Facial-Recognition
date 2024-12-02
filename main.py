import cv2
import numpy as np
import os
import csv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Constants
haar_file = 'haarcascade_frontalface_default.xml'  # Haar Cascade for face detection
datasets = 'datasets'  # Directory containing face image datasets
item_images_path = 'items'  # Directory containing item images

# Email Configuration
EMAIL_ADDRESS = 'abhishekrajooooo79@gmail.com'
EMAIL_PASSWORD = 'qoggbynynwzigqom'
EMAIL_TO = 'deepakrajput4f@gmail.com'

# Initialize the face recognizer
print('Recognizing Face. Please be in sufficient light...')

(images, labels, names, id) = ([], [], {}, 0)

# Load images for training
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Map ID to name
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            image = cv2.imread(path, 0)  # Load image in grayscale
            if image is not None:
                image = cv2.resize(image, (130, 100))  # Resize to fixed dimensions
                images.append(image)
                labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]  # Convert to numpy arrays

# Train the recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Load face detector
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# Load recommendations from CSV
def load_recommendations_csv(file):
    recommendations = {}
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recommendations[row['Customer Name']] = {
                'previous_purchases': row['Previous Purchases'].split(';'),
                'seasonal_winter': row['Seasonal Items (Winter)'].split(';'),
                'seasonal_summer': row['Seasonal Items (Summer)'].split(';')
            }
    return recommendations

recommendations_data = load_recommendations_csv('recommendations.csv')

# Current state variables
current_category = None
current_items = []
click_positions = []

# Handle mouse click events
def on_mouse_click(event, x, y, flags, param):
    global current_category, current_items, click_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 <= y < 130:  # Clicked on "Previous Purchases"
            current_category = 'previous_purchases'
        elif 130 <= y < 210:  # Clicked on "Seasonal Winter"
            current_category = 'seasonal_winter'
        elif 210 <= y < 280:  # Clicked on "Seasonal Summer"
            current_category = 'seasonal_summer'
        else:
            # Check if an item image is clicked
            for pos, item in zip(click_positions, current_items):
                ix, iy, iw, ih = pos
                if ix <= x <= ix + iw and iy <= y <= iy + ih:
                    # Send email notification
                    send_email_notification(name, item)
                    print(f'Request to purchase {item} sent for customer {name}')

# Function to send email notification
def send_email_notification(customer_name, item):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_TO
    msg['Subject'] = f'Purchase Request from {customer_name}'

    body = f'Customer {customer_name} wants to purchase the item: {item}.'
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)  # Replace with your SMTP server and port
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    text = msg.as_string()
    server.sendmail(EMAIL_ADDRESS, EMAIL_TO, text)
    server.quit()

# Main loop for face recognition and interaction
while True:
    _, im = webcam.read()  # Capture frame from webcam
    im = cv2.flip(im, 1)  # Flip image horizontally
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))
        prediction = model.predict(face_resize)

        if prediction[1] < 500:  # Recognized face
            name = names[prediction[0]]
            purchase_history = recommendations_data[name]['previous_purchases']
            seasonal_winter = recommendations_data[name]['seasonal_winter']
            seasonal_summer = recommendations_data[name]['seasonal_summer']
            
            # Display initial welcome message and categories
            welcome_img = np.zeros((800, 800, 3), dtype=np.uint8)
            bg_img = cv2.imread('img.jpg')  # Optional background image
            if bg_img is not None:
                bg_img = cv2.resize(bg_img, (800, 800))
                welcome_img = cv2.addWeighted(welcome_img, 0.3, bg_img, 0.7, 0)

            # Show name and categories
            cv2.putText(welcome_img, f'Welcome back {name}!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(welcome_img, 'Previous Purchases', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(welcome_img, 'Seasonal Winter', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(welcome_img, 'Seasonal Summer', (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # If a category is selected, show items
            if current_category:
                current_items = recommendations_data[name][current_category]
                click_positions = []
                for i, item in enumerate(current_items):
                    item_img_path = os.path.join(item_images_path, f'{item.strip()}.jpg')  # Path to the image file
                    item_img = cv2.imread(item_img_path)
                    if item_img is not None:
                        item_img_resized = cv2.resize(item_img, (200, 200))
                        y_offset = 300 + (i // 3) * 220  # Position in grid
                        x_offset = 50 + (i % 3) * 220
                        click_positions.append((x_offset, y_offset, 200, 200))
                        welcome_img[y_offset:y_offset + 200, x_offset:x_offset + 200] = item_img_resized
                        cv2.putText(welcome_img, item, (x_offset, y_offset + 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("Welcome", welcome_img)
            cv2.setMouseCallback("Welcome", on_mouse_click)  # Set mouse callback
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
