import cv2
from modules.io import count_video_frames, find_file_paths, get_video_frame, load_checkerboard_xml, save_xml
from modules.utils import create_gaussian_model
import numpy as np
import copy
import os

checkerboard_corners_sort = []
checkerboard_corners = set()
current_image = None

def sample_video_frames_intrinsics(video_path, checkerboard_shape, step):
    video = cv2.VideoCapture(video_path)
    frame_samples = []
    frame_iter = 0
    
    while True:
        found_frame, frame = video.read()
        
        if not found_frame:
            break
        elif frame_iter%step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_corners, coords = cv2.findChessboardCorners(gray, checkerboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                                 cv2.CALIB_CB_FAST_CHECK +
                                                                                 cv2.CALIB_CB_FILTER_QUADS)
            if found_corners:
                frame_samples.append(coords)
        
        frame_iter += 1
        
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return frame_samples, (video_height, video_width)

def sample_video_frames_extrinsics(video_path, step):
    video = cv2.VideoCapture(video_path)
    frame_samples = []
    frame_iter = 0
    while True:
        found_frame, frame = video.read()
        if not found_frame:
            break
        elif frame_iter%step == 0:
            frame_samples.append(frame)     
        frame_iter += 1
        
    return np.array(frame_samples), frame_iter

def draw_corners_on_image(image_path, corners, radius=5, color=(0, 0, 255), thickness=-1):
    # Load the image
    image = cv2.imread(image_path)

    # Draw a circle for each corner
    for x, y in corners:
        cv2.circle(image, (x, y), radius, color, thickness)

    return image

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

def find_checkerboard_polygon(frame, background_subtractor):
    """
    This function attempts to locate and approximate the checkerboard polygon containing all the black squares
    :param frame: The frame where the checkerboard polygon should be detected
    :param background_subtractor: The background subtractor used
    :return: The polygon's list of points sorted in clock-wise order
    """
    # Background subtraction
    kernel = np.ones((3, 3), np.uint8)

    foreground_seg = background_subtractor.apply(frame, None, 0.0001)
    f_mask = cv2.erode(foreground_seg, kernel, iterations=1)
    f_mask = cv2.dilate(f_mask, kernel, iterations=1)

    # Foreground noise reduction and histogram equalization
    foreground = cv2.bitwise_and(frame, frame, mask=f_mask)
    foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    foreground = cv2.equalizeHist(foreground_gray)

    # Thresholding phase
    _, f_mask = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(foreground, foreground, mask=f_mask)

    # Find final square mask
    f_mask = cv2.erode(f_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(f_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flat_contours = []
    for ctr in contours:
        flat_contours += [pt[0] for pt in ctr]

    hull = cv2.convexHull(np.array(flat_contours))

    final_mask = np.zeros_like(f_mask)
    cv2.drawContours(final_mask, [hull], -1, 255, -1)

    _, fs_mask = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY_INV)
    squares_mask = cv2.bitwise_and(fs_mask, fs_mask, mask=final_mask)

    squares_mask = cv2.erode(squares_mask, kernel, iterations=1)
    squares_mask = cv2.dilate(squares_mask, kernel, iterations=1)

    # Final rectangle corners search
    lst = []

    for y in range(squares_mask.shape[0]):
        for x in range(squares_mask.shape[1]):
            if squares_mask[y, x]:
                lst.append(np.array([x, y]))

    hull = cv2.convexHull(np.array(lst))

    perimeter = cv2.arcLength(np.array(hull), True)
    approx = cv2.approxPolyDP(np.array(hull), 0.02 * perimeter, True)

    return [p[0] for p in approx]

def corners_selection_callback(event, x, y, flags, params):
    """
    This function is used as a callback during the corners selection phase. When the user presses the left mouse
    button, a new corner point is added. When the user presses the right mouse button, the nearest corner point is
    removed. When the user is done it's sufficient to press a keyboard button to go to the next step.
    :param event: The window event
    :param x: The mouse x location
    :param y: The mouse y location
    :param flags: Not used
    :param params: Not used
    """
    global checkerboard_corners, current_image

    # Local UI update flag
    update = False

    if event == cv2.EVENT_LBUTTONDOWN and len(checkerboard_corners) < 4:
        checkerboard_corners.add((x, y))
        update = True
    elif event == cv2.EVENT_RBUTTONDOWN and len(checkerboard_corners) > 0:
        nearest_point = min(checkerboard_corners,
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners.remove(nearest_point)
        update = True

    if update:
        show_image = copy.deepcopy(current_image)
        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)
        cv2.imshow("Corner selection phase", show_image)

def corners_sorting_callback(event, x, y, flags, params):
    """
    This function is used as a callback during the corners sorting phase. When the user presses the left mouse
    button, the nearest non-selected point is marked as the next point in the ordering. When the user presses the
    right mouse button, the nearest selected point is removed from the ordering.
    :param event: The window event
    :param x: The mouse x location
    :param y: The mouse y location
    :param flags: Not used
    :param params: Not used
    """
    global checkerboard_corners, checkerboard_corners_sort, current_image

    # Local UI update flag
    update = False

    if event == cv2.EVENT_RBUTTONDOWN and len(checkerboard_corners_sort) > 0:
        nearest_point = min(checkerboard_corners_sort,
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sort.pop(checkerboard_corners_sort.index(nearest_point))
        update = True
    elif event == cv2.EVENT_LBUTTONDOWN and len(checkerboard_corners_sort) < 4:
        nearest_point = min(list(filter(lambda p: p not in checkerboard_corners_sort, checkerboard_corners)),
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sort.append(nearest_point)
        update = True

    if update:
        show_image = copy.deepcopy(current_image)
        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)
            if point in checkerboard_corners_sort:
                index = checkerboard_corners_sort.index(point)
                cv2.putText(show_image, "P" + str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2,
                            cv2.LINE_AA)
        cv2.imshow("Corner selection phase", show_image)

def interpolate_and_project_corners(edges, pattern_shape, adjust_outer_edges=False):
    """
    This function computes and returns the interpolated 2D corner points using 4 manually selected board corners.
    :param edges: a list containing 4 manually selected board corners sorted in clockwise order (TP, TR, BR, BL)
    :param pattern_shape: the shape of the checkerboard
    :param adjust_outer_edges: adjusts the checkerboard width and height if set to True. If edges contains outer edges then adjust_outer_edges should be always True
    :return: the list of interpolated 2D checkerboard corners
    """
    if len(edges) != 4:
        raise ValueError("[PROJ_CORNERS]: Edges should always contain 4 different points!")

    if len(pattern_shape) != 2:
        raise ValueError("[PROJ_CORNERS]: Pattern shape should always contain 2 dimensions!")

    # Find the max width and height of the manually selected polygon
    max_width = max(np.linalg.norm(edges[0] - edges[1]),
                    np.linalg.norm(edges[3] - edges[2]))
    max_height = max(np.linalg.norm(edges[1] - edges[2]),
                     np.linalg.norm(edges[3] - edges[0]))

    # Define the mapping coordinates for perspective transform
    output_points = np.float32([[0, 0],
                                [max_width - 1, 0],
                                [max_width - 1, max_height - 1],
                                [0, max_height - 1]])

    # Adjust width and height lengths if specified
    w_adjust, h_adjust = 0, 0

    if adjust_outer_edges:
        w_adjust = max_width / (pattern_shape[1] + 1)
        h_adjust = max_height / (pattern_shape[0] + 1)
        max_width -= 2.0 * w_adjust
        max_height -= 2.0 * h_adjust

    # Compute the inverse perspective transform
    p_matrix = cv2.getPerspectiveTransform(edges.astype(np.float32), output_points)
    inv_p_matrix = np.linalg.inv(p_matrix)

    # Compute the horizontal and vertical step
    w_step = max_width / (pattern_shape[1] - 1)
    h_step = max_height / (pattern_shape[0] - 1)

    projected_corners = []

    # Compute each projected point
    for x in range(0, pattern_shape[1]):
        for y in range(0, pattern_shape[0]):
            point = np.array([w_adjust + x * w_step, h_adjust + y * h_step, 1])
            point = np.matmul(inv_p_matrix, point)
            # Divide each point by its Z component
            point *= (1.0 / point[2])
            # Append only the first 2 elements of each point
            projected_corners.append(point[0:2])

    return projected_corners

def compute_camera_extrinsics(corners, camera_matrix, square_size, pattern_shape):
    default_object_points = np.zeros((pattern_shape[0] * pattern_shape[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2) \
                                   * square_size
                                   
    return cv2.solvePnP(default_object_points, corners, camera_matrix, None)

def compute_camera_intrinsics(image_points, image_shape, checkerboard_pattern, square_size):
    object_points = np.zeros((checkerboard_pattern[0] * checkerboard_pattern[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:checkerboard_pattern[0], 0:checkerboard_pattern[1]].T.reshape(-1, 2) \
                                   * square_size

    object_points = [object_points] * len(image_points)

    re_err, matrix, dist, _, _, std_in, _, _ = \
        cv2.calibrateCameraExtended(object_points, image_points, image_shape, None, None)

    return re_err, matrix, dist, std_in[0:4]

if __name__ == '__main__':
    pattern_shape = (6,8)

    # Find the paths for intrinsic videos
    background_video_files = find_file_paths('data', 'background.avi')
    print(background_video_files)
    
    # Find the paths for extrinsic videos
    foreground_video_files = find_file_paths('data', 'checkerboard.avi')
    print(foreground_video_files)
    
    sampling_step_background = 1
    sampling_step_foreground = 25
    
    checkboard_xml_path = 'data/checkerboard.xml'
    calibrations_path = 'calibrations'
    background_video_path = 'data/cam1/background.avi'
    
    sampling_step = 120

    # Retrieve checkerboard data
    checkerboard_data = load_checkerboard_xml(checkboard_xml_path)

    # Retrieve checkerboard shape and square size
    checkerboard_shape, checkerboard_square_size = (checkerboard_data["CheckerBoardHeight"], checkerboard_data["CheckerBoardWidth"]) \
                                                    ,checkerboard_data["CheckerBoardSquareSize"]                                       
    print("Shape of the checkerboard:", checkerboard_shape)
    
    
    # Find the paths for intrinsic videos
    intrinsics_files = find_file_paths('data', 'intrinsics.avi')
    print(intrinsics_files)
    
    for i in range(len(background_video_files)):
        
        # ---------------------------------------------------- INTRINSICS ---------------------------------------------------- #
        
        print(intrinsics_files[i])
        # Sample the training video for the specified camera
        calibration_images_points, images_shape = sample_video_frames_intrinsics(intrinsics_files[i],
                                                                            checkerboard_shape, sampling_step)
        print("Frames used for intristics calibration:",len(calibration_images_points))
        
        
        # Calibrate the specified camera
        re_err_i, matrix, dist, std_in, = compute_camera_intrinsics(calibration_images_points,
                                                                    images_shape,
                                                                    checkerboard_shape,
                                                                    checkerboard_square_size)
        
        # Log the estimated intrinsics parameters
        # log_camera_intrinsics_confidence(matrix, re_err_i, std_in, "Camera Calibration")
        
        # Save the estimated intrinsics 
        intrinsics_xml_path = os.path.join(calibrations_path, f"intrinsics_cam_{i+1}.xml")
        save_xml(intrinsics_xml_path,
                ["CameraMatrix", "DistortionCoeffs"],
                [matrix, dist])

        
        
        # ---------------------------------------------------- EXTRINSICS ---------------------------------------------------- #
        background_images, total_bg_frames = sample_video_frames_extrinsics(background_video_files[i], sampling_step_background)
        foreground_images, total_fg_frames = sample_video_frames_extrinsics(foreground_video_files[i], sampling_step_foreground)

        print("Background images:",background_images.shape[0])
        print("Foreground images", foreground_images.shape[0])
        
        mean_background, std_background = create_gaussian_model(background_images) 
        mean_foreground, std_foreground = create_gaussian_model(foreground_images) 
        
        cv2.imwrite(f'gaussian/background_mean_cam_{i+1}.jpg', mean_background)
        cv2.imwrite(f'gaussian/background_std_dev_cam_{i+1}.jpg', std_background)
        cv2.imwrite(f'gaussian/foreground_mean_cam_{i+1}.jpg', mean_foreground)
        cv2.imwrite(f'gaussian/foreground_std_dev_cam_{i+1}.jpg', std_foreground)
        
        
        # fetch first video frame
        first_foreground_frame = get_video_frame(foreground_video_files[i], 0)
        knn_subtractor = train_KNN_background_subtractor(background_video_files[i])
        
        # find_checkerboard_polygon with frame
        polygon_points = find_checkerboard_polygon(first_foreground_frame, knn_subtractor)
        checkerboard_corners = {(point[0], point[1]) for point in polygon_points}
        
        # draw polygon points
        current_image = copy.deepcopy(first_foreground_frame)
        show_image = copy.deepcopy(first_foreground_frame)

        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)

        cv2.imshow("Corner selection phase", show_image)
        cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners Selection")
        cv2.setMouseCallback("Corner selection phase", corners_selection_callback)
        
        while True:
            cv2.waitKey(0)
            if len(checkerboard_corners) == 4:
                break

        cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners Sorting")
        cv2.setMouseCallback("Corner selection phase", corners_sorting_callback)

        while True:
            cv2.waitKey(0)
            if len(checkerboard_corners_sort) == 4:
                cv2.setMouseCallback("Corner selection phase", lambda *args: None)
                break
            
        
        
        # compute internal corners
        corners = np.array([np.array([x, y]) for x, y in checkerboard_corners_sort])
        corners = interpolate_and_project_corners(corners, checkerboard_shape, True)
        corners = np.array([[p[0] for p in corners], [p[1] for p in corners]]) \
            .transpose().astype(np.float32)
        
        # Show corners and return
        cv2.drawChessboardCorners(current_image, checkerboard_shape, corners, True)

        cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners")
        cv2.imshow("Corner selection phase", current_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Initialize the sorted corners to zero
        checkerboard_corners_sort = []
        
        print(len(checkerboard_corners))
        
        re_err_e, r_vecs, t_vecs = compute_camera_extrinsics(corners,
                                                            matrix,
                                                            checkerboard_square_size,
                                                            checkerboard_shape)
        
        # Saving the estimated extrinsics to args.camera_path/extrinsics.xml
        extrinsics_xml_path = os.path.join(calibrations_path, f"extrinsics_cam_{i+1}.xml")
        save_xml(extrinsics_xml_path,
                ["RotationVector", "TranslationVector"],
                [r_vecs, t_vecs])    
    
    
        # Final calibration test
        calibration_test_frame = get_video_frame(foreground_video_files[i], 0)
        cv2.drawFrameAxes(calibration_test_frame, matrix, None, r_vecs, t_vecs,
                        checkerboard_square_size * 4, thickness=2)

        # Plotting the calibration frame
        cv2.imshow("Calibration Frame Test", calibration_test_frame)

        # Saving the calibration frame to args.camera_path/calibration_test_frame.jpg
        calibration_test_frame_path = os.path.join(calibrations_path, f"calibration_test_frame_cam_{i+1}.jpg")
        cv2.imwrite(calibration_test_frame_path, calibration_test_frame)

        cv2.waitKey(4000)
        cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    