import cv2

# Load the background image
background = cv2.imread('background.jpg')

# Access the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to access webcam.")
    exit()

# Resize background to match the video frame size
background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green screen (adjust values as needed)
    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask for the person
    mask_inv = cv2.bitwise_not(mask)

    # Segment the person and background using masks
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)

    # Combine foreground and new background
    combined = cv2.add(fg, bg)

    # Show the result
    cv2.imshow('Virtual Background Replacement', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()