import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model("cnp_digit_classifier.h5")
print(model)
global model_used
# Load and preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")      # Convert to grayscale
    img = img.resize((28, 28))                     # Resize to 28x28
    img = np.array(img)
    img = 255 - img                                # Invert if black digit on white
    img = img / 255.0
    #if model_used == load_model("mlp_digit_classifier.h5"):
    #img = img.reshape(1, 28 * 28)

    img = img.reshape(1, 28, 28, 1)
    return img

# Test the image
image_path = "my_digit.jpg"
processed_img = preprocess_image(image_path)

# Predict
pred = model.predict(processed_img)
digit = np.argmax(pred)

print("predict:", pred)
print("Predicted digit:", digit)

# Show image
plt.imshow(processed_img.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {digit}")
plt.show()
