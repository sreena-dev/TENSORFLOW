import cv2 # Import OpenCV

print("--- Starting Webcam for live capture ---")
print("Press 'c' to capture an image and save it.")
print("Press 'q' to quit.")

cap = cv2.VideoCapture(0) # Open the default camera (0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Make sure webcam is connected and not in use.")
    exit() # Exit if camera fails to open

while True:
    ret, frame = cap.read() # Read a frame from the webcam (this is the "live feed" part)

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Display the live feed in a window
    cv2.imshow('Live Webcam Feed (Press "c" to capture, "q" to quit)', frame)

    # Wait for 1ms for a key press (non-blocking)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'): # If 'c' is pressed, capture and save
        print("\n--- Image Captured! Saving as 'captured_image.jpg'... ---")
        # Save the captured frame to a file
        cv2.imwrite('captured_image.jpg', frame)
        print("Image saved.")
        # You could add a brief display of the captured image if you like:
        # cv2.imshow('Captured Image', frame)
        # cv2.waitKey(2000) # Show for 2 seconds
        # cv2.destroyWindow('Captured Image')

    elif key == ord('q'): # If 'q' is pressed, quit the application
        print("Quitting webcam feed.")
        break

# --- Clean up ---
# Release the camera resource
cap.release()
# Destroy all OpenCV windows
cv2.destroyAllWindows()