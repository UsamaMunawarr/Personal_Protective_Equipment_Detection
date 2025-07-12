from ultralytics import YOLO # Import the YOLO class from the ultralytics package
import cv2 # Import OpenCV for image processing
import cvzone # Import cvzone for additional computer vision functionalities
import math # Import math for mathematical operations


######################################
## Webcam Setup and Model Loading ###
######################################
cap = cv2.VideoCapture(2)  # For webcam
cap.set(3, 720)  # Set the width of the webcam feed
cap.set(4, 480)   # Set the height of the webcam feed


######################################
## For Videos Use this code instead ##
########################################
# cap = cv2.VideoCapture(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\05_Person_Protective_Eq_Detector\Personal_Protective_Equipment_Detection\Protective_Eq_Detection.py")# For video file

model = YOLO(r'C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\05_Person_Protective_Eq_Detector\Personal_Protective_Equipment_Detection\ppe.pt')  # Load the YOLOv8 model weights

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
myColor = (0,0,255)

while True:
    success, img = cap.read()  # Read a frame from the webcam
    results = model(img, stream=True)  # Run inference on the webcam feed
    for r in results:
        boxes = r.boxes  # Get the bounding boxes from the results
        for box in boxes:
            ##########################################
            # Get the coordinates of the bounding box
            ##########################################
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            w, h = x2-x1,y2-y1  # Calculate width and height from coordinates

            #############################################
            # if we press control and right click on cornerRect blow chunk it will show the code of 
            # cornerRect and we can change the corner of rectangles accordingly 
            ##############################################
            # cvzone.cornerRect(img, (x1, y1, w, h)) # Draw a rectangle around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)  # Draw a rectangle around the detected object with a pink color
            ################################################
            #### Display Class Name and Confidence Score ###
            ##################################################
            conf = math.ceil((box.conf[0]*100))/100  # Get the confidence score of the detection
            # print(f'Confidence: {conf}')  # Print the confidence score
            # cvzone.putTextRect(img, f'Conf: {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2) # Display the confidence score on the image
            cls = int(box.cls[0])  # Get the class ID of the detected object
            currentClass = classNames[cls]  # Get the class name from the class ID
            if conf> 0.5:  # Only display the class name and confidence score if the confidence is above 0.5
                if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0, 255, 0)
                elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColor = (0, 0, 255)
                else:
                    myColor = (255,0,0)
            ###########################################
            #### Class Name Display on Image ######
            ##########################################
            cls = int(box.cls[0]) # Get the className Variable id suppose 0 == person, 1 == bicycle, etc
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), 
                                max(35, y1)), scale=0.7, thickness=1, colorB=myColor,
                                colorT=(255,255,255),colorR=myColor,offset=5)  # Display the class name and confidence score on the image

    cv2.imshow("Image", img)  # Display the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for a key press for 1 millisecond
        break








##############################################
#### Code to chekc which webcam i am using####
##############################################
# for i in range(3):  # Try indexes 0, 1, and 2 manually
#     print(f"Trying camera index {i}")
#     cap = cv2.VideoCapture(i)
#     if not cap.isOpened():
#         print(f"Camera {i} not available")
#         continue

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Failed to grab frame from camera {i}")
#             break

#         cv2.imshow(f"Camera {i}", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
