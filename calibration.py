import cv2
import numpy as np
import os
import glob
import copy
from modules.io import load_checkerboard_xml, save_xml, find_file_paths, get_video_frame
from modules.subtraction import train_KNN_background_subtractor
from modules.utils import find_checkerboard_polygon, interpolate_and_project_corners


# Global variables
checkerboard_corners = set()
checkerboard_corners_sorted = []
current_image = None


def sample_video_frames(video_path, checkerboard_shape, step):
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


def compute_camera_intrinsics(image_points, image_shape, checkerboard_pattern, square_size):
    object_points = np.zeros((checkerboard_pattern[0] * checkerboard_pattern[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:checkerboard_pattern[0], 0:checkerboard_pattern[1]].T.reshape(-1, 2) \
                                   * square_size

    object_points = [object_points] * len(image_points)

    re_err, matrix, dist, _, _, std_in, _, _ = \
        cv2.calibrateCameraExtended(object_points, image_points, image_shape, None, None)

    return re_err, matrix, dist, std_in[0:4]


def log_camera_intrinsics_confidence(mtx, ret, std, calibration_name="", rounding=3):
    title = "Confidence Of Estimated Camera Parameters" if calibration_name == "" \
        else "[" + calibration_name + "] Confidence Of Estimated Camera Parameters"

    print(title)
    print("Overall RMS Re-Projection Error", round(ret, 3))
    print("Focal Length (Fx)", round(mtx[0][0], rounding), "\tSTD +/-", round(std[0][0], rounding))
    print("Focal Length (Fy)", round(mtx[1][1], rounding), "\tSTD +/-", round(std[1][0], rounding))
    print("Camera Center (Cx)", round(mtx[0][2], rounding), "\tSTD +/-", round(std[2][0], rounding))
    print("Camera Center (Cy)", round(mtx[1][2], rounding), "\tSTD +/-", round(std[3][0], rounding), "\n")


def corners_selection_callback(event, x, y, flags, params):
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
    global checkerboard_corners, checkerboard_corners_sorted, current_image

    # Local UI update flag
    update = False

    if event == cv2.EVENT_RBUTTONDOWN and len(checkerboard_corners_sorted) > 0:
        nearest_point = min(checkerboard_corners_sorted,
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sorted.pop(checkerboard_corners_sorted.index(nearest_point))
        update = True
    elif event == cv2.EVENT_LBUTTONDOWN and len(checkerboard_corners_sorted) < 4:
        nearest_point = min(list(filter(lambda p: p not in checkerboard_corners_sorted, checkerboard_corners)),
                            key=lambda p: np.linalg.norm(np.array([x, y]) - np.array([p[0], p[1]])))
        checkerboard_corners_sorted.append(nearest_point)
        update = True

    if update:
        show_image = copy.deepcopy(current_image)
        for point in checkerboard_corners:
            cv2.circle(show_image, point, 2, (255, 0, 0), -1)
            if point in checkerboard_corners_sorted:
                index = checkerboard_corners_sorted.index(point)
                cv2.putText(show_image, "P" + str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2,
                            cv2.LINE_AA)
        cv2.imshow("Corner selection phase", show_image)


def find_checkerboard_corners(checkerboard_path, background_path, pattern_shape):
    global checkerboard_corners, current_image

    # fetch first video frame
    first_frame = get_video_frame(checkerboard_path, 0)
    knn_subtractor = train_KNN_background_subtractor(background_path)

    # find_checkerboard_polygon with frame
    polygon_points = find_checkerboard_polygon(first_frame, knn_subtractor)
    checkerboard_corners = {(point[0], point[1]) for point in polygon_points}

    # draw polygon points
    current_image = copy.deepcopy(first_frame)
    show_image = copy.deepcopy(first_frame)

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
        if len(checkerboard_corners_sorted) == 4:
            cv2.setMouseCallback("Corner selection phase", lambda *args: None)
            break

    # compute internal corners
    corners = np.array([np.array([x, y]) for x, y in checkerboard_corners_sorted])
    corners = interpolate_and_project_corners(corners, pattern_shape, True)
    corners = np.array([[p[0] for p in corners], [p[1] for p in corners]]) \
        .transpose().astype(np.float32)

    # Show corners and return
    cv2.drawChessboardCorners(current_image, pattern_shape, corners, True)

    cv2.setWindowTitle("Corner selection phase", "Checkerboard Corners")
    cv2.imshow("Corner selection phase", current_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(corners)

def compute_camera_extrinsics(corners, camera_matrix, square_size, pattern_shape):
    default_object_points = np.zeros((pattern_shape[0] * pattern_shape[1], 3), dtype=np.float32)
    default_object_points[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2) \
                                   * square_size
    return cv2.solvePnP(default_object_points, corners, camera_matrix, None)


if __name__ == '__main__':
    
    checkboard_xml_path = 'data/checkerboard.xml'
    # intrinsic_video_path = 'data/cam1/intrinsics.avi'
    calibrations_path = 'calibrations'
    extrisic_video_path = 'data/cam1/checkerboard.avi'
    background_video_path = 'data/cam1/background.avi'
    
    sampling_step = 80

    # Retrieve checkerboard data
    checkerboard_data = load_checkerboard_xml(checkboard_xml_path)

    # Retrieve checkerboard shape and square size
    checkerboard_shape, checkerboard_square_size = (checkerboard_data["CheckerBoardHeight"], checkerboard_data["CheckerBoardWidth"]) \
                                                    ,checkerboard_data["CheckerBoardSquareSize"]                                       
    print("Shape of the checkerboard:", checkerboard_shape)
    
    
    # Find the paths for intrinsic videos
    intrinsics_files = find_file_paths('data', 'intrinsics.avi')
    print(intrinsics_files)
    
    for i, intrinsics_file in enumerate(intrinsics_files):
        print(intrinsics_file)
        # Sample the training video for the specified camera
        calibration_images_points, images_shape = sample_video_frames(intrinsics_file,
                                                                            checkerboard_shape, sampling_step)
        print("Frames used for intristics calibration:",len(calibration_images_points))
        
        
        # Calibrate the specified camera
        re_err_i, matrix, dist, std_in, = compute_camera_intrinsics(calibration_images_points,
                                                                    images_shape,
                                                                    checkerboard_shape,
                                                                    checkerboard_square_size)
        
        # Log the estimated intrinsics parameters
        log_camera_intrinsics_confidence(matrix, re_err_i, std_in, "Camera Calibration")
        
        # Save the estimated intrinsics 
        intrinsics_xml_path = os.path.join(calibrations_path, f"intrinsics_cam_{i+1}.xml")
        save_xml(intrinsics_xml_path,
                ["CameraMatrix", "DistortionCoeffs"],
                [matrix, dist])
        
    
    
    
    print("-"*40)
    
    # Find the paths for extrinsic videos
    extrinsics_files = find_file_paths('data', 'checkerboard.avi')
    print(extrinsics_files)
    
    for i, extrinsics_file in enumerate(extrinsics_files):
        print(extrinsics_file) 
        
        checkerboard_corners = find_checkerboard_corners(extrinsics_file,
                                                        background_video_path,
                                                        checkerboard_shape)
        
        re_err_e, r_vecs, t_vecs = compute_camera_extrinsics(checkerboard_corners,
                                                            matrix,
                                                            checkerboard_data["CheckerBoardSquareSize"],
                                                            checkerboard_shape)
        # Saving the estimated extrinsics to args.camera_path/extrinsics.xml
        extrinsics_xml_path = os.path.join(calibrations_path, f"extrinsics_cam_{i+1}.xml")
        save_xml(extrinsics_xml_path,
                ["RotationVector", "TranslationVector"],
                [r_vecs, t_vecs])
    
    
    # Final calibration test
    calibration_test_frame = get_video_frame(extrinsics_files[0], 0)
    cv2.drawFrameAxes(calibration_test_frame, matrix, None, r_vecs, t_vecs,
                      checkerboard_data["CheckerBoardSquareSize"] * 4, thickness=2)

    # Plotting the calibration frame
    cv2.imshow("Calibration Frame Test", calibration_test_frame)

    # Saving the calibration frame to args.camera_path/calibration_test_frame.jpg
    calibration_test_frame_path = os.path.join(calibrations_path, "calibration_test_frame.jpg")
    cv2.imwrite(calibration_test_frame_path, calibration_test_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)
    
    print("-"*40)
    # Find the paths for background videos
    background_files = find_file_paths('data', 'background.avi')
    print(background_files)
    
    
    
    print("-"*40)
    # Find the paths for intrinsic videos
    reconstruction_video_files = find_file_paths('data', 'video.avi')
    print(reconstruction_video_files)
    
    
    
    