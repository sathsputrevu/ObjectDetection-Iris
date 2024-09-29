from Detector import *
import os

def main():
    videoPath = r"C:\Users\sathw\OneDrive\Desktop\iris\ObjectDetection\tests\video2.mp4"

    # Change these to absolute paths
    configPath = r"C:\Users\sathw\OneDrive\Desktop\iris\ObjectDetection\model_data\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    modelPath = r"C:\Users\sathw\OneDrive\Desktop\iris\ObjectDetection\model_data\frozen_inference_graph.pb"
    classesPath = r"C:\Users\sathw\OneDrive\Desktop\iris\ObjectDetection\model_data\coco.names"

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()
