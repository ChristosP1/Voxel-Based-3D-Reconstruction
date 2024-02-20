import cv2


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