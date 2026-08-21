import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# CONFIG
# -----------------------------------
MODEL_PATH = "trained_model"        # SavedModel directory
IMAGE_PATH = "test/test/PotatoHealthy1.JPG"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -----------------------------------
# Load model
# -----------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# -----------------------------------
# Load & display test image
# -----------------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Test Image")
plt.axis("off")
plt.show()

# -----------------------------------
# Preprocess image
# -----------------------------------
img_resized = cv2.resize(img, IMG_SIZE)
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------------
# Predict
# -----------------------------------
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = np.max(predictions)

predicted_class = CLASS_NAMES[predicted_index]

# -----------------------------------
# Display result
# -----------------------------------
print(f"🦠 Predicted Disease: {predicted_class}")
print(f"📊 Confidence: {confidence:.2%}")

plt.imshow(img)
plt.title(f"{predicted_class}\nConfidence: {confidence:.2%}")
plt.axis("off")
plt.show()
