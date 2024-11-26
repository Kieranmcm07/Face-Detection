# Import necessary libraries
import cv2
import time

# Load the face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if the face detector is loaded correctly
if face_detector.empty():
    print("Error loading face detector")
    exit()

# Open the default camera (index 0)
camera = cv2.VideoCapture(0)

# Initialize variables for calculating FPS
previous_time = 0
current_time = 0

while True:
    # Read a frame from the camera
    success, frame = camera.read()

    # Check if the frame is empty
    if not success:
        print("Error reading frame")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 6)

    # Draw rectangles around the detected faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display the FPS, credit, and quit text
    cv2.putText(
        frame,
        "FPS: " + str(int(fps)),
        (10, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame, "Made by KieranMc", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2
    )
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, 60),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 0, 102),
        2,
    )

    # Display the output
    cv2.imshow("Face Detection", frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
