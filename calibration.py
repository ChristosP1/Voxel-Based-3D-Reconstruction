import cv2
import numpy as np
import os
import copy
from modules.io import load_checkerboard_xml, save_xml

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
    """
    This function logs the estimated value and standard deviation for each intrinsic camera parameter
    :param mtx: camera matrix
    :param ret: re-projection error
    :param std: standard deviation for each estimated intrinsic parameter
    :param calibration_name: calibration name
    :param rounding: number of decimal figures considered while rounding
    """
    title = "Confidence Of Estimated Camera Parameters" if calibration_name == "" \
        else "[" + calibration_name + "] Confidence Of Estimated Camera Parameters"

    print(title)
    print("Overall RMS Re-Projection Error", round(ret, 3))
    print("Focal Length (Fx)", round(mtx[0][0], rounding), "\tSTD +/-", round(std[0][0], rounding))
    print("Focal Length (Fy)", round(mtx[1][1], rounding), "\tSTD +/-", round(std[1][0], rounding))
    print("Camera Center (Cx)", round(mtx[0][2], rounding), "\tSTD +/-", round(std[2][0], rounding))
    print("Camera Center (Cy)", round(mtx[1][2], rounding), "\tSTD +/-", round(std[3][0], rounding), "\n")





if __name__ == '__main__':
    
    checkboard_xml_path = 'data/checkerboard.xml'
    intrinsic_video_path = 'data/cam1/intrinsics.avi'
    calibrations_path = 'calibrations'
    extrisic_video_path = 'data/cam1/checkerboard.avi'
    background_video_path = 'data/cam1/background.avi'
    
    sampling_step = 60

    # Retrieve checkerboard data
    checkerboard_data = load_checkerboard_xml(checkboard_xml_path)

    # Sample the training video for the specified camera
    checkerboard_shape, checkerboard_square_size = (checkerboard_data["CheckerBoardHeight"], checkerboard_data["CheckerBoardWidth"]) \
                                                    ,checkerboard_data["CheckerBoardSquareSize"]
                                                     
    print("Shape of the checkerboard:", checkerboard_shape)
    
    calibration_images_points, images_shape = sample_video_frames(intrinsic_video_path,
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
    intrinsics_xml_path = os.path.join(calibrations_path, "intrinsics.xml")
    save_xml(intrinsics_xml_path,
             ["CameraMatrix", "DistortionCoeffs"],
             [matrix, dist])
    
    
    
    
    
    