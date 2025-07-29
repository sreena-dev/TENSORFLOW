import tensorflow as tf
# from keras.datasets import cifar100 # type: ignore
from keras import Sequential
from keras import layers
import keras
# import cv2
import numpy as np
import matplotlib.pyplot as plt

cifar100=keras.datasets.cifar100

(x_train,y_train),(x_test,y_test) =cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model=Sequential([
    # since it is a image type it uses Flatten 32 is the pixel of this dataset 3==RGB colour scale
    # it is like if you pass in a image in 2D or 3D then you need to pass in as a pixel such that it could be taken into nn
                  layers.Flatten(input_shape=(32, 32, 3)),
                  layers.Dense(units=256,activation='relu'),
                  layers.Dense(units=128,activation='relu'),
                  layers.Dense(units=100,activation='softmax')])

sgd_opt=keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="SGD"
)
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# this indicates that the it is trained with with the training set of datas and then runs for 3 epochs, then validates with test datas
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

pred=model.predict(x_test)


cifar100_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'coffee_cup', 'computer_keyboard', 'couch', 'crab',
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
    'fox', 'girl', 'hamster', 'house', 'kangaroo', 'lamp', 'lawn_mower', 'leopard',
    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# print(cifar100_labels[y_test[0]])
print(cifar100_labels[np.argmax(pred[0])])

for i in range(5):
    plt.grid(False)
    plt.imshow(x_test[i],cmap=plt.cm.binary,interpolation='nearest')
    plt.xlabel("Actual: "+ cifar100_labels[y_test[i][0]])
    plt.title("Prediction: "+cifar100_labels[np.argmax(pred[i])])
# plt.imshow(x_test[7])
plt.tight_layout()
plt.show()

# print("\n--- Starting Webcam for live prediction ---")
# print("Press 'c' to capture and classify the current image.")
# print("Press 'q' to quit.")

# cap = cv2.VideoCapture(0) # Open the default camera (0)

# if not cap.isOpened():
#     print("Error: Could not open video stream. Make sure webcam is connected and not in use.")
#     exit() # Exit if camera fails to open

# while True:
#     ret, frame = cap.read() # Read a frame from the webcam (this is the "live feed" part)

#     if not ret:
#         print("Failed to grab frame. Exiting...")
#         break

#     # Display the live feed
#     cv2.imshow('Live Webcam Feed (Press "c" to capture, "q" to quit)', frame)

#     key = cv2.waitKey(1) & 0xFF # Wait for 1ms for a key press (non-blocking)

#     if key == ord('c'): # If 'c' is pressed, capture and predict
#         print("\n--- Image Captured! Processing... ---")

#         # --- Preprocessing the captured frame ---
#         # 1. OpenCV reads in BGR format, convert to RGB for Keras model
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 2. Resize to 32x32 pixels (CIFAR-100 image size)
#         img_resized = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
#         # 3. Normalize pixel values (0-255 to 0-1)
#         img_normalized = img_resized.astype('float32') / 255.0 # Ensure float32 type
#         # 4. Add batch dimension (1, 32, 32, 3) - Keras models expect batches
#         img_for_prediction = np.expand_dims(img_normalized, axis=0)

#         # --- Make Prediction ---
#         predictions = model.predict(img_for_prediction)
#         # Get the index of the class with the highest probability
#         predicted_class_index = np.argmax(predictions[0])
#         # Get the probability itself for that predicted class
#         predicted_probability = predictions[0][predicted_class_index]

#         predicted_label_name = cifar100_labels[predicted_class_index]

#         print(f"Predicted Class: {predicted_label_name} (Index: {predicted_class_index})")
#         print(f"Predicted Probability: {predicted_probability:.4f}")

#         # Optional: Display the captured image with the prediction overlay
#         # You can draw the prediction text directly on the 'frame'
#         display_frame = frame.copy() # Make a copy to draw on
#         text = f"Pred: {predicted_label_name} ({predicted_probability:.2f})"
#         cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('Captured and Predicted', display_frame)
#         cv2.waitKey(2000) # Show for 2 seconds, then continue live feed
#         cv2.destroyWindow('Captured and Predicted')


#     elif key == ord('q'): # If 'q' is pressed, quit the application
#         print("Quitting webcam feed.")
#         break

# # --- 6. Clean up ---
# # Release the camera resource and destroy all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()