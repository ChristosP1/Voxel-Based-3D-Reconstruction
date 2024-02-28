import cv2
import numpy as np
import os
import tkinter as tk


def show_warning(message_id):
    """
    Shows warnings to the user during execution of the program. Possible warnings include:
    - train_empty for empty training image folder
    - test_empty for empty testing folder
    - images_need_crop for images without equal dimensions
    - image_none for image that fails to be loaded
    - video_none for video that fails to be played
    - incorrect_num_corners for wrong number of chessboard corners
    - no_automatic_corners for automatic corner detection failure
    - no_automatic_corners_online for automatic corner detection failure in online phase
    - no_automatic_corners_online_video for automatic corner detection failure in online phase during video processing
    - approx_corners_sort for instructions to sort approximated corners
    - approx_corners_discard for instructions for manual selection after discarding approximated corners
    - calibration_results_unequal for array length of camera calibration results being unequal

    :param message_id: message ID string to print appropriate message
    """
    # Define the possible messages
    messages = {
        "train_empty": "Calibration image folder is missing the relevant files!",
        "test_empty": "Test folder is missing the relevant files!",
        "images_need_crop": "Not all images have the same dimensions! Images will be cropped!",
        "image_none": "Image could not be loaded and will be skipped!",
        "video_none": "Video could not be played and will be skipped!",
        "incorrect_num_corners": "Incorrect number of corners given!",
        "no_automatic_corners": "Corners not detected automatically! Need to extract manually!\n" +
                                "Select the 4 corners with left clicks and then press any key to continue.\n" +
                                "To undo selections in order use right clicks.",
        "no_automatic_corners_online": "Corners not detected automatically! Image will be discarded from testing!",
        "no_automatic_corners_online_video": "Corners not detected automatically for some frames! Frames were skipped",
        "approx_corners_sort": "Corners not detected automatically! Outer corners have been approximated.\n" +
                               "Select order of the 4 corners with left clicks and then press any key to continue.\n" +
                               "To undo selections in order use right clicks. Closest corner to a click is selected.",
        "approx_corners_discard": "Approximated corners have been discarded and manual extraction is needed!\n" +
                                  "Select the 4 corners with left clicks and then press any key to continue.\n" +
                                  "To undo selections in order use right clicks.",
        "calibration_results_unequal": "Plotting error, array lengths of camera calibration results are not the same!"
    }

    # Fetch the appropriate message
    message = messages.get(message_id, "Unknown Warning")

    # Create the main window
    root = tk.Tk()
    root.title("Warning")

    # Create a label for the message
    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()

    # Start the GUI event loop
    root.mainloop()


def uniform_image_dimensions(directory_path):
    """
    Checks whether all images in given directory have the same width and height dimensions. Crops images to match
    minimum dimensions found if not.

    :param directory_path: directory path
    :return: returns the final shape of the images (original if all the same, otherwise cropped) or None if no images
             present in directory
    """
    # Get list of image file paths
    image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".jpg")]
    if not image_paths:
        return None

    # Initialize variables to store the dimensions
    min_width, min_height = np.inf, np.inf
    dimensions_set = set()

    # First pass to find dimensions and check uniformity
    img = None
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            dimensions_set.add((h, w))
            min_width, min_height = min(min_width, w), min(min_height, h)

    # Check if all images have the same dimensions
    if len(dimensions_set) == 1:
        return img.shape[:2]
    else:
        show_warning("images_need_cropping")

    # Second pass to crop images
    cropped_img = None
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            if h > min_height or w > min_width:
                # Calculate crop dimensions
                top = (h - min_height) // 2
                bottom = h - min_height - top
                left = (w - min_width) // 2
                right = w - min_width - left

                # Crop and save the image
                cropped_img = img[top:h - bottom, left:w - right]
                cv2.imwrite(image_path, cropped_img)

    return cropped_img.shape[:2]


def load_xml_nodes(directory_path, filename, node_tags, node_types=None):
    """
    Load XML file nodes using their tags.

    :param directory_path: directory path
    :param filename: file name
    :param node_tags: array of tags for loaded nodes
    :param node_types: array of types for loaded nodes ("real", "int", "string", or "mat"), if None or unequal length to
                       node_tags then just loads nodes
    :return: dictionary of loaded nodes with their tags as keys
    """
    # Select file to read from
    if not filename.lower().endswith(".xml"):
        filename += ".xml"
    file = cv2.FileStorage(os.path.join(directory_path, filename), cv2.FileStorage_READ)

    # No node types given or unequal node tags and types length, just load nodes
    if node_types is None or len(node_tags) != len(node_types):
        return {node_tag: file.getNode(node_tag) for node_tag in node_tags}

    # Load each node using its tag and its type
    nodes = dict.fromkeys(node_tags)
    for idx, (node_tag, node_type) in enumerate(zip(node_tags, node_types)):
        if node_type == "real":
            nodes[node_tag] = file.getNode(node_tag).real()
        elif node_type == "int":
            nodes[node_tag] = int(file.getNode(node_tag).real())
        elif node_type == "string":
            nodes[node_tag] = file.getNode(node_tag).string()
        elif node_type == "mat":
            nodes[node_tag] = file.getNode(node_tag).mat()
        else:
            nodes[node_tag] = file.getNode(node_tag)

    # Close the file
    file.release()

    return nodes


def save_xml_nodes(directory_path, filename, node_tags, node_values):
    """
    Save XML file nodes with their tags.

    :param directory_path: directory path
    :param filename: file name
    :param node_tags: array of tags for nodes
    :param node_values: array of values of nodes
    """
    # Select file to write to
    if not filename.lower().endswith(".xml"):
        filename += ".xml"
    file = cv2.FileStorage(os.path.join(directory_path, filename), cv2.FileStorage_WRITE)

    # Write every node with its tag
    for node_tag, node_value in zip(node_tags, node_values):
        file.write(node_tag, node_value)

    # Close the file
    file.release()


def get_video_frame(directory_path, filename, frame):
    """
    Gets a specific frame from a video.

    :param directory_path: directory path
    :param filename: file name
    :param frame: frame to return
    :return: returns video frame or None if frame is not found or video doesn't load
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(directory_path, filename))
    if not cap.isOpened():
        return None

    # Loop until all video frames are processed
    frame_count = 0
    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            current_frame = None
            break

        # Specified frame reached
        if frame_count == frame:
            break

        frame_count += 1
    cap.release()

    return current_frame


def get_video_properties(directory_path, filename, fast_frame_count=False):
    """
    Gets dimensions, fps, and frame count properties of a video.

    :param directory_path: directory path
    :param filename: file name
    :param fast_frame_count: if True then gets frame count using cv2.CAP_PROP_FRAME_COUNT which is sometimes inaccurate,
                             otherwise counts frames accurately 1 by 1
    :return: returns video shape (width, height), fps, and frame count
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(directory_path, filename))
    if not cap.isOpened():
        return None

    # Get dimension properties
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_shape = np.array((frame_width, frame_height), dtype=np.int32)

    # Get frame properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Fast but sometimes inaccurate frame count
    if fast_frame_count:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Accurate frame count
    else:
        # Loop until all video frames are processed
        frame_count = 0
        while True:
            # Read video frame
            ret_frame, current_frame = cap.read()
            # Video end
            if not ret_frame:
                break
            frame_count += 1
        cap.release()

    return video_shape, fps, frame_count


