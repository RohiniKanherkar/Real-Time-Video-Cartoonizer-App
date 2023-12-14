# Real-Time Video Cartoonizer

This project utilizes Streamlit and OpenCV to create a real-time video cartoonizer. It applies various image processing techniques to convert live video from your camera into a cartoon-like representation.

## Features

- **Bilateral Filter:** Adjust the strength of the bilateral filter applied to color images.
- **Canny Edge Detection:** Set thresholds for Canny edge detection to enhance the cartoon effect.
- **Erosion:** Control the size of the kernel for eroding contours in the cartoonized output.
- **Rotation:** Rotate the video stream in real-time to achieve different effects.
- **Black and White Mode:** Toggle between color and black-and-white cartoonization.

## Getting Started

To run this project locally, you need to have Python and the required libraries installed. Use the following commands to set up the environment:

```bash
pip install opencv-python numpy streamlit scipy
