from ultralytics import YOLO
import cv2
import os
import math
import cvzone


def ppe_detection(file=None):
    # Validate file path
    if file and not os.path.exists(file):
        print(f"Error: File '{file}' does not exist.")
        return

    # Initialize video capture
    if file is None:
        cap = cv2.VideoCapture(0)  # Webcam
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Load the YOLO model
    model = YOLO("best.pt")

    # Define class names
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)

    while True:
        # Read a frame from the video
        success, img = cap.read()

        # Validate the frame
        if not success or img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Invalid frame or end of video.")
            break

        # YOLO detection
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)

                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)  # Red
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  # Green
                    else:
                        myColor = (255, 0, 0)  # Blue

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # Show the image
        cv2.imshow("PPE Detection", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #file = r"F:\Computer_vision\PPE_detection_YOLO\Videos\ppe-2.mp4"
    #file = r"C:\Users\saiso\OneDrive\Desktop\PPE_detection_Kit-main\Videos\ppe-2.mp4"
    file = r"C:\Users\saiso\OneDrive\Desktop\PPE_detection_Kit-main\Videos\ppe-1.mp4"
    ppe_detection(file)
