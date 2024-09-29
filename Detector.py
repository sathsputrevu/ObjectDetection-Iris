import cv2
import numpy as np
import time
import matplotlib.pyplot as plt  # Added for image display

np.random.seed(20)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def show_image(self, image):
        """ Display the image using matplotlib """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.imshow(image_rgb)
        plt.axis('off')  # Don't show axis
        plt.show()

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video file...")
            return 
        
        success, image = cap.read()
        startTime = 0

        while success:
            currTime = time.time()
            fps = 1 / (currTime - startTime)
            startTime = currTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxsIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxsIdx) != 0:
                for i in range(0, len(bboxsIdx)):
                    bbox = bboxs[np.squeeze(bboxsIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxsIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    # Draw corners
                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    # Top-left
                    cv2.line(image, (x, y), (x + lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=5)

                    # Top-right
                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=5)

                    # Bottom-left
                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x, y + h), (x, y + h - lineWidth), classColor, thickness=5)

                    # Bottom-right
                    cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness=5)

            # Show FPS
            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Display the image using matplotlib
            self.show_image(image)

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()

