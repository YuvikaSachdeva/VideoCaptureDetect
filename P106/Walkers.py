import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier("C:/Users/Yuvika/Desktop/Python/Projects/P106/haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/Yuvika/Desktop/Python/Projects/P106/Vid/walking.avi')

# Loop once the video is successfully loaded
while True:
    
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass each frame to the classifier
    bodies = body_classifier.detectMultiScale(gray)

    for (x, y, w, h) in bodies:
        # Draw a rectangle around the detected area
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if the Space key is pressed
    if cv2.waitKey(1) == 32:  # 32 is the Space Key
        break
cap.release()
cv2.destroyAllWindows()
