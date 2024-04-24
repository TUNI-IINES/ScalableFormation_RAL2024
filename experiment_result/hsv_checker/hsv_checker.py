import cv2
import numpy as np

# SOURCE: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

def nothing(x):
    pass

# Load image
image = cv2.imread('test_frame.jpg')

# Create a window
cv2.namedWindow('image')

# set default value to check
# HSV_min = [ np.array([00, 80, 219], np.uint8) ] # red
# HSV_max = [ np.array([11, 255, 255], np.uint8) ]
# HSV_min = [ np.array([101, 67, 224], np.uint8) ] # blue
# HSV_max = [ np.array([137, 255, 255], np.uint8) ]
# HSV_min = [ np.array([76, 117, 195], np.uint8) ] # green
# HSV_max = [ np.array([90, 255, 255], np.uint8) ]  
HSV_min = np.array([23, 71, 241], np.uint8) # orange
HSV_max = np.array([48, 255, 255], np.uint8)


# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMin', 'image', HSV_min[0])
cv2.setTrackbarPos('SMin', 'image', HSV_min[1])
cv2.setTrackbarPos('VMin', 'image', HSV_min[2])
cv2.setTrackbarPos('HMax', 'image', HSV_max[0])
cv2.setTrackbarPos('SMax', 'image', HSV_max[1])
cv2.setTrackbarPos('VMax', 'image', HSV_max[2])

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()