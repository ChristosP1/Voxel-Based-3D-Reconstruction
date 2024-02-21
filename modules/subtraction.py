import cv2
from modules.io import count_video_frames


def subtract_background(chessboard_image, background_image):
    # Convert both images to grayscale
    gray_chessboard = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    subtracted = cv2.absdiff(gray_chessboard, gray_background)

    # Thresholding to enhance the chessboard
    _, thresh = cv2.threshold(subtracted, 30, 255, cv2.THRESH_BINARY)
    return thresh

def train_KNN_background_subtractor(filepath):
    """
    This function trains a KNN background subtractor using the video located at filepath
    :param filepath: The training video path
    :return: The trained KNN background subtractor
    """
    video = cv2.VideoCapture(filepath)

    subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    while True:
        success, current_frame = video.read()
        if not success:
            break
        subtractor.apply(current_frame, None)

    return subtractor
