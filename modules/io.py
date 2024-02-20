import cv2
import os

def find_file_paths(directory, filename):
    found_files = []

    for folder_path, subfolders, filenames in os.walk(directory):
        for file in filenames:
            if file == filename:
                full_path = os.path.join(folder_path, file)
                found_files.append(full_path)

    return found_files

def load_xml(filepath, tags, custom_process=lambda x: int(x.real())):
    file = cv2.FileStorage(filepath, cv2.FileStorage_READ)
    return {tag: custom_process(file.getNode(tag)) for tag in tags}

def save_xml(filepath, tags, values):
    file = cv2.FileStorage(filepath, cv2.FileStorage_WRITE)
    for tag, value in zip(tags, values):
        file.write(tag, value)
    file.release()

def load_checkerboard_xml(filepath):
    tags = ['CheckerBoardWidth', 'CheckerBoardHeight', 'CheckerBoardSquareSize']
    return load_xml(filepath, tags)

def get_video_frame(filepath, frame_index):
    """
    This function returns the frame at the specified index for the video located at filepath
    :param filepath: The video path
    :param frame_index: The frame index
    :return: The requested frame
    """
    video = cv2.VideoCapture(filepath)
    while True:
        success, current_frame = video.read()
        if not success:
            break
        elif frame_index == 0:
            return current_frame
        frame_index -= 1
    return None

def count_video_frames(filepath):
    """
    This function counts the number of frames in the video located at filepath
    :param filepath: The video path
    :return: The number of frames
    """
    frame_count = 0
    video = cv2.VideoCapture(filepath)
    while True:
        success, _ = video.read()
        if not success:
            break
        frame_count += 1
    return frame_count