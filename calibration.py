import cv2
import numpy as np
import os
import glob
from copy import deepcopy
from modules.io import load_checkerboard_xml, save_xml, find_file_paths, get_video_frame
from modules.subtraction import subtract_background
from modules.utils import interpolate_points_from_manual_corners


# Global variables
checkerboard_corners = set()
checkerboard_corners_sorted = []
current_image = None

manual_corners = []
images_temp = []


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

        # Add text near click
        cv2.putText(current_image, str(len(manual_corners)), manual_corners[-1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # Draw circle at click
        cv2.circle(current_image, manual_corners[-1],
                   radius=3, color=(0, 255, 0), thickness=-1)

        # Connect corners with line
        if len(manual_corners) > 1:
            cv2.line(current_image, manual_corners[-2], manual_corners[-1],
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if len(manual_corners) == 4:
            cv2.line(current_image, manual_corners[0], manual_corners[-1],
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

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
        

def extract_image_points(checkerboard_shape, file_path, more_exact_corners=False,
                         result_time_visible=1000):

    global manual_corners, images_temp

    # Get the first frame from the video since the checkerboard is not moving
    image = get_video_frame(file_path, 0)
    
    # Try to detect the chessboard corners automatically using grayscale image and all flags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                               cv2.CALIB_CB_FILTER_QUADS +
                                                                               cv2.CALIB_CB_FAST_CHECK)

    # No automatic corner detection, going to manual corner selection
    if not ret:

        # Corner selection window, calls callback function
        images_temp.append(deepcopy(image))
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", image)
        cv2.setMouseCallback("Manual Corners Selection (Press any key when all corners selected)",
                                 manual_corner_selection)
        # Loop until 4 corners selected and any key is pressed
        while True:
            cv2.waitKey(0)
            if len(manual_corners) == 4:
                cv2.destroyAllWindows()
                break

        print(manual_corners)
        
        # Corner interpolation using selected corners
        corners = interpolate_points_from_manual_corners(manual_corners, checkerboard_shape)
        # corners = [tuple(row[0]) for row in corners]
        print(corners)
        # Reset parameters used for manual corner detection and go back to original image
        image = images_temp[0]
        images_temp = []
        manual_corners = []

    # Increase corner accuracy with more exact corner positions
    if more_exact_corners:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))


    # Draw extracted corners on image
    cv2.drawChessboardCorners(image, checkerboard_shape, corners, True)

    # Show results of extraction
    cv2.imshow("Extracted Chessboard Corners", image)
    cv2.waitKey(result_time_visible)
    cv2.destroyAllWindows()

    
    return corners



def compute_camera_extrinsics(corners, camera_matrix, square_size, pattern_shape):
    default_object_points = np.zeros((pattern_shape[0] * pattern_shape[1], 3), dtype=np.float32)
    print(default_object_points.shape)
    default_object_points[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2) \
                                   * square_size
                                   
    return cv2.solvePnP(default_object_points, corners, camera_matrix, None)


if __name__ == '__main__':
    
    checkboard_xml_path = 'data/checkerboard.xml'
    # intrinsic_video_path = 'data/cam1/intrinsics.avi'
    calibrations_path = 'calibrations'
    extrisic_video_path = 'data/cam1/checkerboard.avi'
    background_video_path = 'data/cam1/background.avi'
    
    sampling_step = 120

    # Retrieve checkerboard data
    checkerboard_data = load_checkerboard_xml(checkboard_xml_path)

    # Retrieve checkerboard shape and square size
    checkerboard_shape, checkerboard_square_size = (checkerboard_data["CheckerBoardHeight"], checkerboard_data["CheckerBoardWidth"]) \
                                                    ,checkerboard_data["CheckerBoardSquareSize"]                                       
    print("Shape of the checkerboard:", checkerboard_shape)
    
    
    
    # ------------------------------------------------- INTRINSICS ------------------------------------------------- #
    # Find the paths for intrinsic videos
    intrinsics_files = find_file_paths('data', 'intrinsics.avi')
    print(intrinsics_files)
    
    # Find the paths for extrinsic videos
    extrinsics_files = find_file_paths('data', 'checkerboard.avi')
    print(extrinsics_files)
    
    for i in range(len(intrinsics_files)):
        print(intrinsics_files[i])
        # Sample the training video for the specified camera
        calibration_images_points, images_shape = sample_video_frames(intrinsics_files[i],
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
        
    
        # ------------------------------------------------- EXTRINSICS ------------------------------------------------- #
        print("-"*100)
    
        print(extrinsics_files[i]) 
        
        checkerboard_corners = extract_image_points(checkerboard_shape, extrinsics_files[i])
                            
        re_err_e, r_vecs, t_vecs = compute_camera_extrinsics(checkerboard_corners,
                                                            matrix,
                                                            checkerboard_square_size,
                                                            checkerboard_shape)
        # Saving the estimated extrinsics to args.camera_path/extrinsics.xml
        extrinsics_xml_path = os.path.join(calibrations_path, f"extrinsics_cam_{i+1}.xml")
        save_xml(extrinsics_xml_path,
                ["RotationVector", "TranslationVector"],
                [r_vecs, t_vecs])
        print("Extrinsic error:", re_err_e, r_vecs, t_vecs)
    
    
        # Final calibration test
        calibration_test_frame = get_video_frame(extrinsics_files[0], 0)
        cv2.drawFrameAxes(calibration_test_frame, matrix, None, r_vecs, t_vecs,
                        checkerboard_square_size * 4, thickness=2)

        # Plotting the calibration frame
        cv2.imshow("Calibration Frame Test", calibration_test_frame)

        # Saving the calibration frame to args.camera_path/calibration_test_frame.jpg
        calibration_test_frame_path = os.path.join(calibrations_path, "calibration_test_frame.jpg")
        cv2.imwrite(calibration_test_frame_path, calibration_test_frame)

        cv2.waitKey(4000)
        cv2.destroyAllWindows()
    
    
    
    
    # print("-"*40)
    # # Find the paths for background videos
    # background_files = find_file_paths('data', 'background.avi')
    # print(background_files)
    
    
    
    # print("-"*40)
    # # Find the paths for intrinsic videos
    # reconstruction_video_files = find_file_paths('data', 'video.avi')
    # print(reconstruction_video_files)
    
    
    
    