# cv2 -> library for face detection( stands for computer vision)
import cv2

#Loading pre-trained data on face frontals(haarcascade_frontalface_default in this case) from github/opencv/opencv/tree/map/data/haarcascade
# trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose a image to detect face of
# img=cv2.imread('ave.jpg')

#Must convert to grayscale for the algorithm to work
# grayscaled_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #COLOR_BR2GRAY is taken form the haarcascade_frontalface_default.xml model which converts the color to grey

#Detect Faces
# face_cordinates= trained_face_data.detectMultiScale(grayscaled_img)     # detectMultiScale - detects objects of different sizes in the input image. returhs coordinates of the rectangl surrounding the image

# print(face_cordinates)      #prints the list of cordinates of the rectangel surrounding the image.

#Draw rectangle around the face
# (x, y, w, h)= face_cordinates[0]    # x, y represents the coordinates of the lower left coordinate of the rectangle, w- width of the rectangle and h is the height of the rectangle

#for a single image in a file
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)     # (x,y) are the coordinates of the lower left point of the rectangle and we add w to h and h to y to obtain the upper right coordinate of the rectangle.
# (0, 255, 0) is the touple representing the color scheme RGB respectively. Here both red and blue are turned off and only green is displayed on the rectangle.
# the last 2 is the thickness of the rectangle surrounding the face

# For a multi image file
# for (x,y,w,h) in face_cordinates:
    # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

# cv2.imshow('Face Detector', img)
# cv2.waitKey()       # prevents the image to open in the terminal and close in split second

'''--------------------------------------------------------------------------------------------------------------------------------------------------'''

''' DETECTING FACES IN REAL TIME (USING THE WEBCAM)'''

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam= cv2.VideoCapture(0)     # captures the video from the webcam live. Here 0 stands for webcam. A path to a video file can also be provided instead of 0

# since the video is a continuous series of frames, we will have to rum a while loop, for the programme to run through, till the video is running
while True:
    # Read the current frame
    succesful_frame_read, frame= webcam.read()

    #Must convert to grayscale
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_cordinates= trained_face_data.detectMultiScale(grayscaled_img)

    # for (x,y,w,h) in face_cordinates:
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)


    cv2.imshow('Face Detector', frame)
    key= cv2.waitKey(1)      # the waitkey waits till 1 milliseconds before jumping to next frame

    if key==81 or key==113:
        break

#Rekease the video capture object
webcam.release()

 
     