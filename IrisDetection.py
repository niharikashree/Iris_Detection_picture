import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect the iris in an eye region
def detect_iris(eye):
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    
    # Detect circles in the eye region using Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_eye, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=30, 
        minRadius=5, 
        maxRadius=30
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(eye, (x, y), r, (0, 255, 0), 2)  # Draw the outer circle (iris)
            cv2.circle(eye, (x, y), 2, (0, 0, 255), 3)  # Draw the center of the iris
        return True
    return False

# Main video capture
cap = cv2.VideoCapture(0)
people_detected = 0
iris_detected_people = 0
iris_detected_complete = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    people_detected = len(faces)  # Update count of people based on faces detected

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        
        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey + eh, ex:ex + ew]
            
            # Detect and draw the iris, update count if iris is detected
            if detect_iris(eye):
                iris_detected_people += 1
                break  # Exit the eye loop after detecting iris for this person
    
    # Check if the iris detection is complete for all people
    if iris_detected_people == people_detected and people_detected > 0:
        iris_detected_complete = True
        break
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit the loop by pressing the 'ESC' key
    if cv2.waitKey(1) == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Output the final count of people after detecting their irises
if iris_detected_complete:
    print(f"Total number of people with detected irises: {people_detected}")
else:
    print("Iris detection was not successful for all people.")
