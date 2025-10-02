# week1_pose_test.py

import cv2
import mediapipe as mp

# --------------------------
# 1. Setup MediaPipe
# --------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Use static_image_mode=True since we're working with images
pose = mp_pose.Pose(static_image_mode=True)

# --------------------------
# 2. Load your image
# --------------------------
# Replace 'squat.jpg' with your own image file
image = cv2.imread("squat.png")
if image is None:
    print("Error: Image not found. Make sure the file path is correct.")
    exit()

# Convert BGR to RGB for MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------------------------
# 3. Process image
# --------------------------
results = pose.process(image_rgb)

# --------------------------
# 4. Draw skeleton if landmarks detected
# --------------------------
annotated_image = image.copy()

if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # Print coordinates of some key landmarks (optional)
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

else:
    print("No person detected in the image.")

# --------------------------
# 5. Display the annotated image
# --------------------------
cv2.imshow("Skeleton Overlay", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
