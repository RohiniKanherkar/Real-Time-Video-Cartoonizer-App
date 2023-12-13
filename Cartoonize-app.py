import cv2
import numpy as np
import streamlit as st
from collections import defaultdict
from scipy import stats
import base64

# Functions for cartoonization

def update_c(C, hist):
    while True:
        groups = defaultdict(list)

        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))

        if np.sum(new_C - C) == 0:
            break
        C = new_C

    return C, groups

def K_histogram(hist):
    alpha = 0.001
    N = 80
    C = np.array([128])

    while True:
        C, groups = update_c(C, hist)

        new_C = set()
        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue

            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (C[i] + left) / 2
                    c2 = (C[i] + right) / 2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C


def caart(img, bilateral_filter_value, canny_threshold1, canny_threshold2, erode_kernel_size):
    kernel = np.ones((2, 2), np.uint8)
    output = np.array(img)

    if len(output.shape) == 3:  # Check if the image is in color
        x, y, c = output.shape
    else:  # Grayscale image
        x, y = output.shape
        c = 1

    if c == 3:  # Apply bilateral filter for color images
        for i in range(c):
            output[:, :, i] = cv2.bilateralFilter(output[:, :, i], bilateral_filter_value, 150, 150)
    else:  # Skip bilateral filter for grayscale images
        output = cv2.bilateralFilter(output, bilateral_filter_value, 150, 150)

    edge = cv2.Canny(output, canny_threshold1, canny_threshold2)
    
    if c == 3:  # Convert to HSV for color images
        output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []

    if c == 3:
        hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
        hists.append(hist)
        hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
        hists.append(hist)
        hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
        hists.append(hist)
    else:
        hist, _ = np.histogram(output, bins=np.arange(256 + 1))
        hists.append(hist)

    C = []
    for h in hists:
        C.append(K_histogram(h))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))

    if c == 3:  # Convert back to RGB for color images
        output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    for i in range(c):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=erode_kernel_size)

    return output



def apply_black_and_white(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_rotation(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def main():
    bg_container = st.container()
    with bg_container:
        # Set the background image using custom CSS
        st.markdown(
            """
            <style>
            .stApp {
                
                background-image: url('https://images.unsplash.com/photo-1563089145-599997674d42?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');  
                background-size: cover;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    st.title('Real-Time Video Cartoonizer')

    video_capture_object = cv2.VideoCapture(0)
    out = None  # VideoWriter object

    st.sidebar.title("Cartoonize Parameters")
    bilateral_filter_value = st.sidebar.slider("Bilateral Filter Value", 5, 50, 9)
    canny_threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
    canny_threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
    erode_kernel_size = st.sidebar.slider("Erode Kernel Size", 1, 5, 2)
    rotation_angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
    black_and_white = st.sidebar.checkbox("Black and White")
    
    start_stop_button = st.button("Start/Stop Video")

    stframe = st.empty()

    while start_stop_button:
        ret, frame = video_capture_object.read()
        if not ret:
            break

        if out is None:
            # Initialize VideoWriter if not done yet
            out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))

        if black_and_white:
            frame = apply_black_and_white(frame)

        if rotation_angle != 0:
            frame = apply_rotation(frame, rotation_angle)

        cartoon_frame = caart(frame, bilateral_filter_value, canny_threshold1, canny_threshold2, erode_kernel_size)

        out.write(cartoon_frame)

        # Display the cartoonized frame
        stframe.image(cartoon_frame, channels="BGR" if len(cartoon_frame.shape) == 3 and cartoon_frame.shape[2] == 3 else "GRAY")

    # Release resources when the loop is stopped
    if out is not None:
        out.release()
    video_capture_object.release()

    # Add a download button for the recorded video
    st.markdown("## Download Recorded Video")
    download_button = st.button("Download Video")

    if download_button:
        st.markdown(get_binary_file_downloader_html('out.mp4', 'Video'), unsafe_allow_html=True)


# Function to generate a download link for the recorded video
def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/mp4;base64,{b64}" download="{file_path}">Download {file_label}</a>'
    return href

if __name__ == '__main__':
    main()
