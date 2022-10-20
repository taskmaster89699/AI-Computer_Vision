#import open cv
import cv2
image_file='cars1.jpg'    #Insert image to detect cars present in it

# Get the pre-trained car classifier
classifier_file='cars.xml'

# create open cv image
img= cv2.imread(image_file)

# Create a car classifier
car_tracker= cv2.CascadeClassifier(classifier_file)

# Convert the image to grayscale
black_and_white= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the cars
cars= car_tracker.detectMultiScale(black_and_white)



# Draw the rectangles around the cars detected
for (x,y,w,h) in cars:      # x and y are the cordinates of the lower left corner. w and h are width and height
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the cars spotted
cv2.imshow('Car Detector', img)

# Don't autoclose (wait for the image to close)
cv2.waitKey()
