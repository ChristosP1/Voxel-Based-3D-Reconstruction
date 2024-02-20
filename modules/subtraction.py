import cv2
from modules.io import count_video_frames


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


def train_hsv_KNN_subtractor(filepath, threshold):
    """
        This function trains a KNN background subtractor using the hsv version of the video located at filepath
        :param filepath: The training video path
        :param threshold: The foreground threshold
        :return: The trained KNN background subtractor
    """
    subtractor = cv2.createBackgroundSubtractorKNN(history=count_video_frames(filepath),
                                                   detectShadows=False,
                                                   dist2Threshold=threshold)
    video = cv2.VideoCapture(filepath)

    while True:
        success, frame = video.read()
        if not success:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        subtractor.apply(hsv)

    return subtractor
