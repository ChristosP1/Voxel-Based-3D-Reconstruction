import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import background_subtraction
import utils

# Global variables for manual corner detection callback
manual_corners = []
approx_corners = []
images_temp = []


def load_chessboard_info(chessboard_input_path="data", chessboard_input_filename="checkerboard.xml"):
    """
    Loads chessboard shape and square size from XML file.

    :param chessboard_input_path: chessboard xml file directory path
    :param chessboard_input_filename: chessboard xml file name
    :return: returns chessboard number of intersection points horizontally and vertically (horizontal, vertical),
             chessboard square size in mm
    """
    # Select tags for loaded nodes and their types
    node_tags = ["CheckerBoardWidth", "CheckerBoardHeight", "CheckerBoardSquareSize"]
    node_types = ["int" for _ in range(len(node_tags))]

    # Load nodes
    nodes = utils.load_xml_nodes(chessboard_input_path, chessboard_input_filename, node_tags, node_types)

    # Format outputs into shape and square size
    chessboard_shape = (nodes.get("CheckerBoardWidth"), nodes.get("CheckerBoardHeight"))
    chessboard_square_size = nodes.get("CheckerBoardSquareSize")

    return chessboard_shape, chessboard_square_size


def manual_corner_selection(event, x, y, flags, param):
    """
    Callback function to capture user's clicks during manual corner detection. Left-clicking places a corner on the
    window and right-clicking removes the last placed corner. Labels with numbering will be placed at every placed
    corner. When more than 1 corner is placed then the corners will be connected by a line in order of placement.

    :param event: type of event
    :param x: X coordinate of event (click)
    :param y: Y coordinate of event (click)
    :param flags: not used in application
    :param param: not used in application
    """
    global manual_corners, images_temp

    # Left click to add a corner if not all 4 placed
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_corners) < 4:
        # Add corner to list of corners
        manual_corners.append([x, y])

        # Copy last frame to make changes to it
        current_image = deepcopy(images_temp[-1])

        # Draw circle at click
        cv2.circle(current_image, manual_corners[-1],
                   radius=3, color=(255, 0, 0), thickness=-1)

        # Connect corners with line
        if len(manual_corners) > 1:
            cv2.line(current_image, manual_corners[-2], manual_corners[-1],
                     color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        if len(manual_corners) == 4:
            cv2.line(current_image, manual_corners[0], manual_corners[-1],
                     color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # Add number text near click
        cv2.putText(current_image, str(len(manual_corners)), manual_corners[-1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # Save new frame
        images_temp.append(current_image)
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", current_image)
    # Right click to remove a corner if one already is placed
    elif event == cv2.EVENT_RBUTTONDOWN and len(manual_corners) > 0:
        # Removed last placed corner and return to previous frame
        manual_corners.pop()
        images_temp.pop()
        current_image = images_temp[-1]
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", current_image)


def manual_corner_sorting(event, x, y, flags, param):
    """
    Callback function to capture user's clicks during manual corner sorting. Left-clicking selects nearest corner on the
    window and right-clicking removes the last selection. Labels with numbering will be placed at every selected corner.

    :param event: type of event
    :param x: X coordinate of event (click)
    :param y: Y coordinate of event (click)
    :param flags: not used in application
    :param param: not used in application
    """
    global manual_corners, images_temp

    # Left click to select a corner if not all 4 selected
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_corners) < 4:
        # Add corner to list of corners

        # Calculate distances between the given point and each point in the approximated corners
        distances = np.linalg.norm(approx_corners - np.array([x, y]), axis=1)
        nearest_corner_index = np.argmin(distances)

        # Remove nearest corner from list and add it to selected corners
        nearest_corner = approx_corners.pop(nearest_corner_index)
        manual_corners.append(nearest_corner)

        # Copy last frame to make changes to it
        current_image = deepcopy(images_temp[-1])

        # Add number text near click
        cv2.putText(current_image, str(len(manual_corners)), manual_corners[-1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # Save new frame
        images_temp.append(current_image)
        cv2.imshow("Corner Sorting (Press any key when all corners sorted)", current_image)
    # Right click to deselect a corner if one already is selected
    elif event == cv2.EVENT_RBUTTONDOWN and len(manual_corners) > 0:
        # Remove last selected corner, add it back to available corners, and return to previous frame
        deselected_corner = manual_corners.pop()
        approx_corners.append(deselected_corner)
        images_temp.pop()
        current_image = images_temp[-1]
        cv2.imshow("Corner Sorting (Press any key when all corners sorted)", current_image)


def sort_corners_clockwise(corners, origin="top-left"):
    """
    Sorts given corner coordinates in clockwise order.

    :param corners: array of corner points ([x, y])
    :param origin: which corner point starts the clockwise order (bottom-left, top-left, top-right, or bottom-right)
    :return: returns array of sorted corners
    """
    # Calculate the centroid of the corners
    centroid = np.mean(corners, axis=0)

    # Sort corners and determine their relative position to the centroid
    top = sorted([corner for corner in corners if corner[1] < centroid[1]], key=lambda point: point[0])
    bottom = sorted([corner for corner in corners if corner[1] >= centroid[1]], key=lambda point: point[0],
                    reverse=True)

    # Sort top and bottom corners depending on first element
    if origin == "top-left":
        return np.array(top + bottom, dtype="float32")
    elif origin == "top-right":
        return np.array([top[1]] + bottom + [top[0]], dtype="float32")
    elif origin == "bottom-right":
        return np.array(bottom + top, dtype="float32")
    else:
        return np.array([bottom[1]] + top + [bottom[0]], dtype="float32")


def interpolate_image_points_from_corners(corners, chessboard_shape, sort_corners=False, corners_are_outer=False,
                                          change_point_order=False):
    """
    Interpolates chessboard points from the 4 corners using homogenous coordinates and perspective transform for
    enhanced accuracy over a flat interpolation.

    :param corners: array of 4 corner points ([x, y])
    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param sort_corners: if True then sorts corners clockwise starting with top-left corner, otherwise keeps given order
    :param corners_are_outer: if True then corners are outer corners and dimensions should be adjusted to find the inner
                              corners, otherwise corners are considered inner corners and no adjustment is necessary
    :param change_point_order: if False then point order will start at top-left and end on bottom-right moving through
                               columns before changing rows as already saved, if True then point order will start from
                               bottom-left and end on top-right moving through rows before changing column
    :return: returns array of 2D interpolated image points or None if wrong number of corners given (unequal to 4)
    """
    if len(corners) != 4:
        utils.show_warning("incorrect_num_corners")
        return None

    # Sort corners to (top-left, top-right, bottom-right, bottom-left)
    if sort_corners:
        corners = sort_corners_clockwise(corners, origin="top-left")
    else:
        corners = np.array(corners, dtype="float32")

    # Calculate the maximum width and height
    max_width = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[2]))
    max_height = max(np.linalg.norm(corners[2] - corners[1]), np.linalg.norm(corners[3] - corners[0]))

    # Use maximum width and height to form destination coordinates for perspective transform
    dest_corners = np.float32([[0, 0], [max_width - 1, 0],
                               [max_width - 1, max_height - 1], [0, max_height - 1]])

    # Adjustment for cases where the corners given are outer corners
    horizontal_adjust = 0
    vertical_adjust = 0
    if corners_are_outer:
        horizontal_adjust = max_width / (chessboard_shape[0] + 1)
        vertical_adjust = max_height / (chessboard_shape[1] + 1)
        max_width -= 2 * horizontal_adjust
        max_height -= 2 * vertical_adjust

    # Horizontal and vertical step calculation using chessboard shape
    horizontal_step = max_width / (chessboard_shape[0] - 1)
    vertical_step = max_height / (chessboard_shape[1] - 1)

    interpolated_row = []
    interpolated_points = []
    # Perform perspective transform for accuracy improvement
    p_matrix = cv2.getPerspectiveTransform(corners, dest_corners)

    # Get inverse matrix for projecting points from the transformed space back to the original image space
    inverted_p_matrix = np.linalg.inv(p_matrix)

    # Compute each projected point
    for y in range(0, chessboard_shape[1]):
        for x in range(0, chessboard_shape[0]):
            # Calculate the position of the current point relative to the grid using homogenous coordinates
            point = np.array([horizontal_adjust + x * horizontal_step,
                              vertical_adjust + y * vertical_step,
                              1])

            # Multiply with inverse matrix to project point from transformed space back to original image space
            point = np.matmul(inverted_p_matrix, point)

            # Divide point by its Z
            point /= point[2]

            # Append the X and Y of point to the list of interpolated points in row
            interpolated_row.append(point[:2])
        # Append interpolated points in row to interpolated points
        interpolated_points.append(interpolated_row)
        interpolated_row = []

    # If change_point_order is True then point order will start from bottom-left and end on top-right
    # moving through rows before changing column
    # if False then point order will start at top-left and end on bottom-right
    # moving through columns before changing row as already saved
    interpolated_points = np.array(interpolated_points, dtype="float32")
    if change_point_order:
        interpolated_points = np.flip(interpolated_points, axis=0)
        interpolated_points = np.transpose(interpolated_points, (1, 0, 2))

    # Return (MxN, 1, 2) array to match automatic corner detection output
    return np.reshape(interpolated_points, (-1, 1, 2))


def extract_corners(image, bg_model):
    """
    Approximates chessboard outer corners by background subtraction, thresholding, and morphological operations.

    :param image: BGR image
    :param bg_model: trained background model
    :return: returns array with the 4 approximated corners
    """
    # Apply background subtraction model to subtract background
    bg_model_mask = bg_model.apply(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Erode and then dilate (opening) to remove noise
    bg_model_mask = cv2.morphologyEx(bg_model_mask, cv2.MORPH_OPEN, kernel)

    # Get chessboard region using the foreground mask from the background model and improve contrast
    chessboard = cv2.bitwise_and(image, image, mask=bg_model_mask)
    chessboard_gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    chessboard_gray = cv2.equalizeHist(chessboard_gray)

    # Keep only white part in chessboard region
    _, chessboard_binary_white = cv2.threshold(chessboard_gray, 120, 255, cv2.THRESH_BINARY)

    # Get black square contours
    contours, hierarchy = cv2.findContours(chessboard_binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract convex hull (form a polygon from contours)
    contours_hull = cv2.convexHull(np.concatenate(contours))
    # Draw convex hull (chessboard without the excess white around it)
    chessboard_cropped = np.zeros_like(chessboard_binary_white)
    cv2.drawContours(chessboard_cropped, [contours_hull], -1, 255, -1)

    # Keep only black part in cropped chessboard region with a stricter threshold
    _, chessboard_binary_black = cv2.threshold(chessboard_gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_squares = cv2.bitwise_and(chessboard_binary_black, chessboard_binary_black, mask=chessboard_cropped)
    # Erode to straighten squares
    black_squares = cv2.erode(black_squares, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # Keep black points (marked white) as (x, y)
    y, x = np.where(black_squares == 255)
    black_points = np.column_stack((x, y))
    # Extract convex hull (form a polygon from points)
    black_points_hull = cv2.convexHull(black_points)
    perimeter = cv2.arcLength(black_points_hull, True)
    # Find corners
    corners = cv2.approxPolyDP(black_points_hull, 0.02 * perimeter, True)

    return corners.reshape(-1, 2)


def extract_corners_and_interpolate_image_points(image, chessboard_shape, bg_video_input_path="data/cam",
                                                 bg_video_input_filename="background.avi"):
    """
    Approximates chessboard outer corners and presents to user with instructions for manual sorting. User sorts the
    approximated corners by clicking on a prompt window of the image with the corners to select the nearest unselected
    corner. After all 4 corners have been selected the user is shown the results of extracting image points using
    the approximated corners and can discard or accept them. If the image points are rejected, the user is given
    instructions to perform manual corner selection by clicking on a prompt window of the image.

    :param image: BGR image
    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param bg_video_input_path: training background video directory path
    :param bg_video_input_filename: training background video file name (including extension)
    :return: returns array of extracted 2D chessboard image points, array of approximated 2D chessboard outer corners,
             and an image with approximated outer corners and interpolated inner corners if user doesn't discard
             the approximated corners, otherwise the last 2 are set to None

    """
    global images_temp, manual_corners, approx_corners

    # Give frame count of video as history for background model training
    _, _, frame_count = utils.get_video_properties(bg_video_input_path, bg_video_input_filename)
    # Train background model
    bg_model_knn = background_subtraction.train_KNN_background_model(bg_video_input_path, bg_video_input_filename,
                                                                     use_hsv=False, history=frame_count,
                                                                     dist_threshold=425, detect_shadows=False)

    # Approximate 4 outer corners
    outer_corners = extract_corners(image, bg_model_knn)

    # Draw corners on image
    images_temp.append(deepcopy(image))
    for corner in outer_corners:
        cv2.circle(images_temp[0], (corner[0], corner[1]), 2, (0, 0, 255), -1)
    approx_corners = deepcopy(outer_corners).tolist()

    utils.show_warning("approx_corners_sort")
    cv2.imshow("Corner Sorting (Press any key when all corners sorted)", images_temp[0])
    cv2.setMouseCallback("Corner Sorting (Press any key when all corners sorted)", manual_corner_sorting)
    # Loop until 4 corners selected and any key is pressed
    while True:
        cv2.waitKey(0)
        if len(manual_corners) == 4:
            cv2.destroyAllWindows()
            break

    # Corner interpolation using sorted corners
    corners = interpolate_image_points_from_corners(manual_corners, chessboard_shape, corners_are_outer=True)

    # Draw extracted corners on image
    approx_image = deepcopy(images_temp[0])
    #cv2.drawChessboardCorners(approx_image, chessboard_shape, corners, True)
    for corner in corners:
        point = corner[0]
        x = int(point[0])
        y = int(point[1])
        cv2.circle(approx_image, (x, y), 2, (255, 0, 0), -1)

    # Show results of extraction
    cv2.imshow("Extracted Chessboard Corners (Press any key to keep or ESC to discard)", approx_image)
    # ESC to discard the corners and do manual selection
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        outer_corners = None
        approx_image = None

        # Reset parameters used for manual corner detection
        images_temp = []
        manual_corners = []
        approx_corners = []

        # Corner selection window, calls callback function
        images_temp.append(deepcopy(image))
        utils.show_warning("approx_corners_discard")
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", image)
        cv2.setMouseCallback("Manual Corners Selection (Press any key when all corners selected)",
                             manual_corner_selection)
        # Loop until 4 corners placed and any key is pressed
        while True:
            cv2.waitKey(0)
            if len(manual_corners) == 4:
                cv2.destroyAllWindows()
                break

        # Corner interpolation using selected corners
        corners = interpolate_image_points_from_corners(manual_corners, chessboard_shape)

    # Reset parameters used for manual corner detection
    images_temp = []
    manual_corners = []
    approx_corners = []

    cv2.destroyAllWindows()

    return corners, outer_corners, approx_image


def extract_image_points_from_video(chessboard_shape, video_input_path="data/cam",
                                    video_input_filename="intrinsics.avi", frame_interval=50, stop_frame=-1,
                                    more_exact_corners=True, result_time_visible=1000,
                                    handle_manual_corners=False, bg_video_input_filename="background.avi",
                                    output_manual_frame=False,
                                    output_manual_frame_filename="checkerboard_imagepoints.jpg",
                                    output_video=False, output_video_filename="intrinsics_imagepoints.mp4"):
    """
    Parses frames of given video and detects chessboard corners used for camera calibration.

    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param video_input_path: training video directory path
    :param video_input_filename: training video file name (including extension)
    :param frame_interval: video frames to skip (default 50 to keep 1 frame per second of a 50 fps video)
    :param stop_frame: video frame to stop the process once reached, -1 to not stop process
    :param more_exact_corners: increases corner accuracy with more exact corner positions if set to True, otherwise
                               keeps original corner positions (not applied when automatic corner detection fails)
    :param result_time_visible: milliseconds to keep result of corner extraction for a frame to screen, 0 to wait for
                                key press, -1 to not show result to screen
    :param handle_manual_corners: if True then when automatic corner detection fails the user performs a process
                                  to select corners manually, by sorting approximations or overriding them, otherwise
                                  frames where automatic corner detection fails are discarded
    :param bg_video_input_filename: training background video file name (including extension) in the same path as
                                    video_input_path, used for corner approximation if handle_manual_corners is True
    :param output_manual_frame: if True then outputs the first processed frame where automatic corner detection
                                failed and corner approximation was used and kept by the user in the same path as
                                video_input_path, used when handle_manual_corners is True
    :param output_manual_frame_filename: output frame file name (including extension), used when output_manual_frame is
                                         True
    :param output_video: if True then outputs a video with kept frames with automatic corner detections (1 fps) in the
                         same path as video_input_path
    :param output_video_filename: output video file name (including extension), used when output_video is True
    :return: returns array of 2D chessboard image points for every kept frame and the shape of the video
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(video_input_path, video_input_filename))
    if not cap.isOpened():
        utils.show_warning("video_none")
        return None, None

    # Get dimensions for video
    video_shape, _, _ = utils.get_video_properties(video_input_path, video_input_filename, True)

    # Define the codec for MP4 and create VideoWriter object for output video
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if not os.path.isdir(video_input_path):
            os.makedirs(video_input_path)
        output_cap = cv2.VideoWriter(os.path.join(video_input_path, output_video_filename), fourcc, 1, video_shape)
    else:
        output_cap = None

    # Loop until all video frames are processed
    image_points = []
    frame_count = 0
    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            break

        # Stop frame reached, stop process
        if frame_count == stop_frame and stop_frame != -1:
            break

        # Check if frame will be used according to interval
        if frame_count % frame_interval == 0:
            # For manual corner selection return
            ret_manual = False

            # Try to detect the chessboard corners automatically using grayscale frame and all flags
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                                   cv2.CALIB_CB_FILTER_QUADS +
                                                                                   cv2.CALIB_CB_FAST_CHECK)

            # Additionally handling cases of manual corner selection if automatic detection failed
            if not ret and handle_manual_corners:
                corners, _, approx_frame = \
                    extract_corners_and_interpolate_image_points(current_frame, chessboard_shape,
                                                                 bg_video_input_path=video_input_path,
                                                                 bg_video_input_filename=bg_video_input_filename)

                # Output first frame with approximated corners
                if approx_frame is not None and output_manual_frame:
                    cv2.imwrite(os.path.join(video_input_path, output_manual_frame_filename), approx_frame)
                    output_manual_frame = False

                # Set ret to True as we extracted corners
                ret_manual = True

            # Automatic detection succeeded or got corners from manual corner selection
            if ret or ret_manual:
                # Increase corner accuracy with more exact corner positions
                if ret and more_exact_corners:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

                # Store extracted image corners
                image_points.append(corners)

                # Draw extracted corners on image
                cv2.drawChessboardCorners(current_frame, chessboard_shape, corners, True)

                # Show results of extraction
                if result_time_visible != -1:
                    cv2.imshow("Extracted Chessboard Corners", current_frame)
                    cv2.waitKey(result_time_visible)
                    cv2.destroyAllWindows()

                if output_video:
                    output_cap.write(current_frame)

        frame_count += 1

    cap.release()
    if output_video:
        output_cap.release()

    cv2.destroyAllWindows()

    return image_points, video_shape


def discard_bad_image_points(image_points, image_shape, chessboard_shape, chessboard_square_size=1,
                             discard_threshold=0.15):
    """
    Discards image points from images that make calibration performance worse. Calibrates camera using all image points
    and uses the mean returned reprojection error as baseline. Then iteratively goes through all possible combinations
    by excluding 1 image's points from the image points set to compare against all image points used. If keeping that
    image's points makes the calibration significantly worse given a threshold, then those points are discarded.

    :param image_points: array of 2D chessboard image points for every image
    :param image_shape: shape of the images
    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param chessboard_square_size: chessboard square size in mm
    :param discard_threshold: threshold of error improvement when an image's points are excluded to discard them
    :return: returns array of 2D chessboard image points for every kept image, array of kept image point indexes,
             array of 2D chessboard image points for every discarded image, and array of discarded image point indexes
    """
    # Calibrate camera using all image points and use the mean returned reprojection error as baseline
    baseline_ret, _, _, _, _, _, _, _ = estimate_intrinsic_camera_parameters(image_points, image_shape,
                                                                             chessboard_shape, chessboard_square_size,
                                                                             print_results=False)

    # Go through all possible combinations by excluding 1 image from image set to compare against all images used
    kept_image_points = []
    kept_image_points_idx = []
    discarded_image_points = []
    discarded_image_points_idx = []
    for i in range(len(image_points)):
        image_points_excluding_1 = deepcopy(image_points)
        excluded = image_points_excluding_1.pop(i)
        ret, _, _, _, _, _, _, _ = estimate_intrinsic_camera_parameters(image_points_excluding_1, image_shape,
                                                                        chessboard_shape, chessboard_square_size,
                                                                        print_results=False)

        # Made performance worse given a threshold means image points will be discarded, otherwise kept
        if baseline_ret - ret >= discard_threshold:
            discarded_image_points.append(excluded)
            discarded_image_points_idx.append(i)
        else:
            kept_image_points.append(excluded)
            kept_image_points_idx.append(i)

    return kept_image_points, kept_image_points_idx, discarded_image_points, discarded_image_points_idx


def estimate_intrinsic_camera_parameters(image_points, image_shape, chessboard_shape, chessboard_square_size=1,
                                         print_results=True):
    """
    Estimates intrinsic camera parameters using extracted image points and calculated object points.

    :param image_points: array of 2D chessboard image points for every image
    :param image_shape: shape of the images
    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param chessboard_square_size: chessboard square size in mm
    :param print_results: prints results of calibration if True
    :return: returns mean reprojection error, camera matrix, distortion coefficients, rotation vector, translation
             vector, standard deviation over intrinsic parameters, standard deviation over extrinsic parameters,
             per view reprojection error
    """
    # Prepare object points [[0,0,0], [1,0,0], [2,0,0] ...,[chessboard_horizontal, chessboard_vertical,0]]
    # Multiply by chessboard square size in mm, keep X, Y (Z=0, stationary XY plane chessboard)
    object_points = np.zeros((np.prod(chessboard_shape), 3), np.float32)
    object_points[:, :2] = np.indices(chessboard_shape).T.reshape(-1, 2) * chessboard_square_size
    #object_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), dtype=np.float32)
    #object_points[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) \
    #                       * chessboard_square_size

    # Repeat object points for every image
    object_points = [object_points for _ in range(len(image_points))]

    # Calibrate camera using object points and image points
    # Get mean reprojection error, camera matrix, distortion coefficients, rotation and translation vectors
    # Extended function also gives std deviation over intrinsic parameters and extrinsic parameters, per view error
    err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err = cv2.calibrateCameraExtended(object_points,
                                                                                             image_points,
                                                                                             image_shape,
                                                                                             None, None)
    # Print results of calibration
    if print_results:
        print("Camera Matrix:")
        print(mtx)
        print("Mean Reprojection Error:", f"{err:.2f}")
        print("Focal Length Fx:", f"{mtx[0][0]:.2f} ± {in_std[0][0]:.2f}")
        print("Focal Length Fy:", f"{mtx[1][1]:.2f} ± {in_std[1][0]:.2f}")
        print("Center Point Cx:", f"{mtx[0][2]:.2f} ± {in_std[2][0]:.2f}")
        print("Center Point Cy:", f"{mtx[1][2]:.2f} ± {in_std[3][0]:.2f}")
        print()

    return err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err


def plot_intrinsic_calibration_results(errs, per_view_errs, mtxs, in_stds, plot_output_path="plots",
                                       plot_output_filename="intrinsic_params_runs_comparison.png"):
    """
    Plots results of intrinsic camera calibration. Supports multiple calibration runs.

    :param errs: array of mean reprojection errors (1 position for every run)
    :param per_view_errs: array of per view reprojection errors (1 position for every run)
    :param mtxs: array of camera matrices (1 position for every run)
    :param in_stds: array of standard deviations over intrinsic parameters (1 position for every run)
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    if len(errs) != len(per_view_errs) != len(mtxs) != len(in_stds):
        utils.show_warning("calibration_results_unequal")
        return

    # Extract intrinsic parameters
    fx = [mtx[0][0] for mtx in mtxs]
    fy = [mtx[1][1] for mtx in mtxs]
    cx = [mtx[0][2] for mtx in mtxs]
    cy = [mtx[1][2] for mtx in mtxs]

    # Extract standard deviations for intrinsic parameters
    fx_stds = [in_std[0][0] for in_std in in_stds]
    fy_stds = [in_std[1][0] for in_std in in_stds]
    cx_stds = [in_std[2][0] for in_std in in_stds]
    cy_stds = [in_std[3][0] for in_std in in_stds]

    # Create calibration run labels
    calibration_runs = ["Calibration " + str(i + 1) for i in range(len(errs))]

    # Plot with 3 rows 2 columns
    fig, ax = plt.subplots(3, 2, figsize=(16, 16))

    # Colormap for plots
    cmap = plt.get_cmap("brg")
    colors = cmap(np.linspace(0, 1, len(errs)))

    # Plot mean reprojection error
    ax[0, 0].set_title("Mean Reprojection Error")
    ax[0, 0].bar(calibration_runs, errs, color=colors, label=[f"{err:.2f}" for err in errs])
    ax[0, 0].legend()

    # Plot reprojection error for every image
    ax[0, 1].set_title("Per View Reprojection Error")
    for run in range(len(errs)):
        ax[0, 1].scatter([str(i) for i in range(len(per_view_errs[run]))], per_view_errs[run],
                         color=colors[run], label=calibration_runs[run])
    ax[0, 1].tick_params(axis="x", rotation=90)
    ax[0, 1].legend()

    # Plot focal length Fx
    ax[1, 0].set_title("Focal Length Fx")
    labels = [f"{focal_length:.2f} ± {std:.2f}" for focal_length, std in zip(fx, fx_stds)]
    ax[1, 0].set_title("Focal Length Fx")
    for run in range(len(errs)):
        ax[1, 0].errorbar(calibration_runs[run], fx[run], yerr=fx_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[1, 0].legend(loc="upper center")

    # Plot focal length Fy
    ax[1, 1].set_title("Focal Length Fy")
    labels = [f"{focal_length:.2f} ± {std:.2f}" for focal_length, std in zip(fy, fy_stds)]
    for run in range(len(errs)):
        ax[1, 1].errorbar(calibration_runs[run], fy[run], yerr=fy_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[1, 1].legend(loc="upper center")

    # Plot center point Cx
    ax[2, 0].set_title("Center Point Cx")
    labels = [f"{center:.2f} ± {std:.2f}" for center, std in zip(cx, cx_stds)]
    for run in range(len(errs)):
        ax[2, 0].errorbar(calibration_runs[run], cx[run], yerr=cx_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[2, 0].legend(loc="upper center")

    # Plot center point Cy
    ax[2, 1].set_title("Center Point Cy")
    labels = [f"{center:.2f} ± {std:.2f}" for center, std in zip(cy, cy_stds)]
    for run in range(len(errs)):
        ax[2, 1].errorbar(calibration_runs[run], cy[run], yerr=cy_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[2, 1].legend(loc="upper center")

    # Adjust layout
    plt.tight_layout()

    # Output plot to file
    if not os.path.isdir(plot_output_path):
        os.makedirs(plot_output_path)
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()


def estimate_extrinsic_camera_parameters(mtx, dist, image_points, chessboard_shape, chessboard_square_size=1,
                                         pnp_ransac=False, print_results=True):
    """
    Estimates extrinsic camera parameters using camera matrix, extracted image points, and calculated object points.

    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param image_points: array of 2D chessboard image points
    :param chessboard_shape: chessboard number of intersection points horizontally and vertically (horizontal, vertical)
    :param chessboard_square_size: chessboard square size in mm
    :param pnp_ransac: if True then incorporates RANSAC algorithm for more robust estimation for Perspective-n-Point
                       (PnP) to find rotation and translation vectors
    :param print_results: prints results of translation if True
    :return: The estimated rotation vector and translation vector or None and None if estimation failed
    """
    # Prepare object points [[0,0,0], [1,0,0], [2,0,0] ...,[chessboard_horizontal, chessboard_vertical,0]]
    # Multiply by chessboard square size in mm, keep X, Y (Z=0, stationary XY plane chessboard)
    object_points = np.zeros((np.prod(chessboard_shape), 3), np.float32)
    object_points[:, :2] = np.indices(chessboard_shape).T.reshape(-1, 2) * chessboard_square_size
    #object_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), dtype=np.float32)
    #object_points[:, :2] = np.mgrid[0:chessboard_shape[1], 0:chessboard_shape[0]].T.reshape(-1, 2) \
    #                       * chessboard_square_size

    # Find the rotation and translation vectors
    # Incorporate RANSAC algorithm for more robust estimation
    if pnp_ransac:
        ret, rvecs, tvecs, _ = cv2.solvePnPRansac(object_points, image_points, mtx, dist)
    else:
        ret, rvecs, tvecs = cv2.solvePnP(object_points, image_points, mtx, dist)

    # Failed to estimate
    if not ret:
        return None, None

    # Print results of estimation
    if print_results:
        print("Rotation vector:")
        print(rvecs)
        print("Translation vector:")
        print(tvecs)
        print()

    return rvecs, tvecs


def draw_axes_on_chessboard(image, image_points, mtx, dist, rvecs, tvecs, chessboard_square_size=1,
                            chessboard_square_span=1):
    """
    Draws 3D axes with their center at the origin of the world coordinates.

    :param image: image to draw on
    :param image_points: array of 2D chessboard image points to take first corner from as axes center
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param chessboard_square_size: chessboard square size in mm
    :param chessboard_square_span: chessboard number of squares each axis spans across
    """
    # Axes span a number of chessboard squares
    axis_length = chessboard_square_size * chessboard_square_span
    # 3 axes corners with negative Z value so axis faces the camera
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) * axis_length

    # Project 3D points to image plane
    image_points_axes, _ = cv2.projectPoints(axes, rvecs, tvecs, mtx, dist)

    # First chessboard corner as axes center
    origin_corner = image_points[0][0]

    # Convert to integer for drawing
    image_points_axes = image_points_axes.astype(np.int32).reshape(-1, 2)
    origin_corner = origin_corner.astype(np.int32)

    # Draw red arrow for horizontal axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[0].ravel()), color=(0, 0, 255), thickness=2)
    # Draw green arrow for vertical axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[1].ravel()), color=(0, 255, 0), thickness=2)
    # Draw blue arrow for Z axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[2].ravel()), color=(255, 0, 0), thickness=2)


def draw_cube_on_chessboard(image, mtx, dist, rvecs, tvecs, chessboard_square_size=1, chessboard_square_span=1):
    """
    Draws a cube at the origin of the world coordinates.

    :param image: image to draw on
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param chessboard_square_size: chessboard square size in mm
    :param chessboard_square_span: chessboard number of squares the cube base sides span across
    """
    # Cube sides span a number of chessboard squares
    cube_side_length = chessboard_square_size * chessboard_square_span
    # 8 cube corners with negative Z value so cube faces the camera
    cube = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]) * cube_side_length

    # Project 3D points to image plane
    image_points_cube, _ = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
    # Convert to integer for drawing
    image_points_cube = image_points_cube.astype(np.int32).reshape(-1, 2)

    # Draw cube floor
    cv2.drawContours(image, [image_points_cube[:4]], -1, color=(130, 249, 255), thickness=2)

    # Draw cube pillars
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(image, tuple(image_points_cube[i]), tuple(image_points_cube[j]), color=(130, 249, 255), thickness=2)

    # Draw cube top
    cv2.drawContours(image, [image_points_cube[4:]], -1, color=(130, 249, 255), thickness=2)


def test_camera_parameters_with_image(mtx, dist, rvecs, tvecs, image, image_points, chessboard_square_size=1,
                                      draw_axes=True, draw_cube=False, result_time_visible=-1, output_result=False,
                                      output_result_path="data/cam", output_result_filename="test.jpg"):
    """
    Tests camera parameters by drawing items at the origin of the world coordinates. Depending on the parameters passed
    to the function, 3D axes with their center at the origin or a cube at the origin can be drawn. To do this, the first
    frame of the selected video is used with the corresponding extracted image points (if given, otherwise extracted).

    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param image: image to draw on
    :param image_points: array of 2D chessboard image points
    :param chessboard_square_size: chessboard square size in mm
    :param result_time_visible: milliseconds to keep result of corner extraction for an image to screen, 0 to wait for
                                key press, -1 to not show result to screen
    :param output_result: if True then outputs the frame with drawings on it in the same path as video_input_path
    :param output_result_path: testing output directory path, used when output_result is True
    :param output_result_filename: testing output frame file name (including extension), used when output_result is True
    :return:
    """
    # Draw axes that span 3 squares on chessboard
    if draw_axes:
        # Draw with negative Z value so axis faces the camera
        draw_axes_on_chessboard(image, image_points, mtx, dist, rvecs, tvecs, chessboard_square_size, 3)
        # Draw with positive Z value so axis faces away from the camera
        #cv2.drawFrameAxes(frame, mtx, dist, rvecs, tvecs, length=3*chessboard_square_size, thickness=2)
    # Draw cube that spans 2 squares on chessboard
    if draw_cube:
        draw_cube_on_chessboard(image, mtx, dist, rvecs, tvecs, chessboard_square_size, 2)

    # Show results
    if result_time_visible != -1:
        cv2.imshow("Camera Parameter Test", image)
        cv2.waitKey(result_time_visible)
        cv2.destroyAllWindows()

    # Output result
    if output_result:
        cv2.imwrite(os.path.join(output_result_path, output_result_filename), image)


if __name__ == '__main__':
    # Directories
    data_path = "data"
    plots_path = "plots"
    cam1_path = os.path.join(data_path, "cam1")
    cam2_path = os.path.join(data_path, "cam2")
    cam3_path = os.path.join(data_path, "cam3")
    cam4_path = os.path.join(data_path, "cam4")
    cam_paths = [cam1_path, cam2_path, cam3_path, cam4_path]

    # Load chessboard info
    chessboard_shape, chessboard_square_size = load_chessboard_info(data_path, "checkerboard.xml")
    print("Loaded chessboard info from file:")
    print("Shape:", chessboard_shape)
    print("Square Size in mm:", chessboard_square_size)
    print()

    # Run calibrations for every camera
    errs = []
    per_view_errs = []
    mtxs = []
    in_stds = []
    # Frames discarded from each camera calibration
    # Indexes were found by running discarding function in comments, but are saved for speed purposes here
    discarded_frames_idx = [[9, 31, 35], [14], [], [65]]
    for camera in range(1, len(cam_paths)+1):
        print("\n=======================INTRINSICS=======================")
        print("Extracting image points from video frames for camera " + str(camera) + ".")
        image_points, video_shape = extract_image_points_from_video(chessboard_shape, cam_paths[camera-1],
                                                                    "intrinsics.avi", frame_interval=50,
                                                                    more_exact_corners=True, result_time_visible=-1,
                                                                    output_video=True)

        print("Running intrinsic calibration for camera " + str(camera) +
              " with " + str(len(image_points)) + " frames.")
        err, mtx, dist, rvecs, tvecs,\
            in_std, ex_std, per_view_err = estimate_intrinsic_camera_parameters(image_points, video_shape,
                                                                                chessboard_shape,
                                                                                chessboard_square_size,
                                                                                print_results=True)

        """
        # Discarding bad image points
        print("Discarding frames with bad calibration performance.")
        kept_image_points, kept_image_points_idx, \
            discarded_image_points, discarded_image_points_idx = discard_bad_image_points(image_points,
                                                                                          video_shape,
                                                                                          chessboard_shape,
                                                                                          chessboard_square_size,
                                                                                          discard_threshold=0.05)
        """

        # Skipping discarding function for speed, use already calculated kept frames
        # Comment out if re-running discarding function
        kept_image_points = deepcopy(image_points)
        for idx in sorted(discarded_frames_idx[camera-1], reverse=True):
            kept_image_points.pop(idx)
        kept_image_points_idx = [idx for idx in range(len(image_points)) if idx not in discarded_frames_idx[camera-1]]

        # Run calibration with kept image points
        print("Running intrinsic calibration for camera " + str(camera) +
              " with " + str(len(kept_image_points)) + "/" + str(len(image_points)) + " frames after discarding.")
        err_d, mtx_d, dist_d, rvecs_d, tvecs_d,\
            in_std_d, ex_std_d, per_view_err_d = estimate_intrinsic_camera_parameters(kept_image_points, video_shape,
                                                                                      chessboard_shape,
                                                                                      chessboard_square_size,
                                                                                      print_results=True)

        # Map None to per view error where image points were removed from the original frames for plotting
        per_view_err_map = [[None] for _ in range(len(per_view_err))]
        for idx, value in zip(kept_image_points_idx, per_view_err_d):
            per_view_err_map[idx] = value

        # Plot intrinsic calibration results with and without discarded image points
        plot_intrinsic_calibration_results([err, err_d], [per_view_err, per_view_err_map], [mtx, mtx_d],
                                           [in_std, in_std_d], plots_path,
                                           plot_output_filename="intrinsic_params_discard_bad_images_camera_"
                                                                + str(camera) + ".png")

        # Save camera results with kept image points
        errs.append(err_d)
        per_view_errs.append(per_view_err_d)
        mtxs.append(mtx_d)
        in_stds.append(in_std_d)

        print("=======================EXTRINSICS=======================")
        print("Extracting image points from video frame for camera " + str(camera) + ".")
        image_points, _ = \
            extract_image_points_from_video(chessboard_shape, cam_paths[camera-1], "checkerboard.avi",
                                            frame_interval=50, stop_frame=50, more_exact_corners=True,
                                            result_time_visible=-1, handle_manual_corners=True,
                                            output_manual_frame=True)

        print("Running extrinsic calibration for camera " + str(camera) + ".")
        rvecs, tvecs = estimate_extrinsic_camera_parameters(mtx, dist, image_points[0], chessboard_shape,
                                                            chessboard_square_size, print_results=True)

        # Test by drawing on the chessboard
        print("Evaluating calibration for camera " + str(camera) + " by drawing on the video frame.")
        test_frame = utils.get_video_frame(cam_paths[camera - 1], "checkerboard.avi", 0)
        test_camera_parameters_with_image(mtx, dist, rvecs, tvecs, test_frame, image_points[0], chessboard_square_size,
                                          draw_axes=True, draw_cube=False, result_time_visible=5000, output_result=True,
                                          output_result_path=cam_paths[camera-1], output_result_filename="test.jpg")

        # Output intrinsics and extrinsics to config file
        utils.save_xml_nodes(cam_paths[camera-1], "config.xml",
                             ["CameraMatrix", "DistortionCoeffs", "RotationVector", "TranslationVector"],
                             [mtx, dist, rvecs, tvecs])

    # Plot intrinsic calibration results from all cameras
    plot_intrinsic_calibration_results(errs, per_view_errs, mtxs, in_stds, plots_path,
                                       plot_output_filename="intrinsic_params_cameras_comparison.png")
