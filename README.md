# Object Detection with YOLO and SSD Models

## Overview

This project implements real-time object detection. It processes video streams and applies object detection algorithms to display bounding boxes around detected objects in real time.

## Features

- Real-time object detection using YOLO and SSD models.
- Detects objects from video files.
- Displays bounding boxes with object labels.
- Utilizes pre-trained models for quick and accurate detection.
  
## Tech Stack

- **Programming Language:** Python
- **Libraries/Frameworks:** OpenCV, TensorFlow, Keras, NumPy
- **Models:** YOLOv3, SSD MobileNet
- **Tools:** Pre-trained models for object detection

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sathsputrevu/objectDetection-Iris
    cd object-detection
    ```

2. Install the required dependencies manually:

    ```bash
    pip install opencv-python
    pip install tensorflow
    pip install keras
    pip install numpy
    ```

3. Move the downloaded weights and config files into the `model_data` directory.

## Usage

1. Run the main script for object detection:

    ```bash
    python model_data/main.py
    ```

2. To process specific video files, place your videos in the `tests` directory and modify the script as needed to load the correct file path.

    ```bash
    python model_data/main.py --input tests/video2.mp4
    ```

3. You can switch between YOLO and SSD by adjusting the configurations in `main.py`.

## File Structure

```
object-detection/
│
├── model_data/
│   ├── __pycache__/
│   ├── coco.names
│   ├── Detector.py
│   ├── frozen_inference_graph.pb
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   └── main.py
│
├── tests/
│   ├── cars_in_highway.mp4
│   ├── video2.mp4
│   └── video3.mp4
│
├── README.md
```

## Model Details

- **YOLO (You Only Look Once):** A fast and accurate object detection model that processes the entire image in a single forward pass.
- **SSD MobileNet:** A lightweight object detection model with good performance for mobile and real-time applications.

## Future Improvements

- Add additional object detection models.
- Integrate with custom datasets for specific object detection needs.
- Enhance real-time detection speed through further optimization.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License.
