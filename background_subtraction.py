import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import utils


def train_KNN_background_model(bg_video_input_path="data/cam", bg_video_input_filename="background.avi", use_hsv=True,
                               history=500, dist_threshold=400.0, detect_shadows=True, learning_rate=-1):
    """
    Trains a KNN (K-Nearest Neighbors) background model on a video.

    :param bg_video_input_path: training background video directory path
    :param bg_video_input_filename: training background video file name (including extension)
    :param use_hsv: converts video frames from BGR to HSV if True
    :param history: length of history, determines the number of previous frames that are considered
    :param dist_threshold: threshold on the squared distance between the pixel and the sample to decide whether a pixel
                           is close to that sample
    :param detect_shadows: If true, the background model will detect shadows and mark them as -1
    :param learning_rate: learning rate for background model to learn background (-1 for automatic)
    :return: returns trained KNN background model
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(bg_video_input_path, bg_video_input_filename))
    if not cap.isOpened():
        return None

    background_model = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist_threshold,
                                                         detectShadows=detect_shadows)

    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            break

        # Convert to HSV
        if use_hsv:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Apply background subtractor on frame
        background_model.apply(current_frame, None, learning_rate)

    return background_model


def train_MOG_background_model(bg_video_input_path="data/cam", bg_video_input_filename="background.avi", use_hsv=True,
                               history=200, n_mixtures=5, bg_ratio=0.7, noise_sigma=0, learning_rate=-1):
    """
    Trains a MOG (Mixture of Gaussians) background model on a video.

    :param bg_video_input_path: training background video directory path
    :param bg_video_input_filename: training background video file name (including extension)
    :param use_hsv: converts video frames from BGR to HSV if True
    :param history: length of history, determines the number of previous frames that are considered
    :param n_mixtures: number of Gaussian mixtures used to model each pixel
    :param bg_ratio: specifies the ratio of the number of pixels in the background to the total number of pixels
                     in the image
    :param noise_sigma: specifies the standard deviation of the Gaussian noise added to the pixel intensity distribution
    :param learning_rate: learning rate for background model to learn background (-1 for automatic)
    :return: returns trained MOG background model
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(bg_video_input_path, bg_video_input_filename))
    if not cap.isOpened():
        return None

    background_model = cv2.bgsegm.createBackgroundSubtractorMOG(history=history, nmixtures=n_mixtures,
                                                                backgroundRatio=bg_ratio, noiseSigma=noise_sigma)

    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            break

        # Convert to HSV
        if use_hsv:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Apply background subtractor on frame
        background_model.apply(current_frame, None, learning_rate)

    return background_model


def train_MOG2_background_model(bg_video_input_path="data/cam", bg_video_input_filename="background.avi", use_hsv=True,
                                history=500, var_threshold=16, detect_shadows=True, learning_rate=-1):
    """
    Trains a MOG2 (Mixture of Gaussians Version 2) background model on a video.

    :param bg_video_input_path: training background video directory path
    :param bg_video_input_filename: training background video file name (including extension)
    :param use_hsv: converts video frames from BGR to HSV if True
    :param history: length of history, determines the number of previous frames that are considered
    :param var_threshold: threshold on the squared Mahalanobis distance between the pixel and the model to decide
                          whether a pixel is well described by the background model
    :param detect_shadows: If true, the background model will detect shadows and mark them as -1
    :param learning_rate: learning rate for background model to learn background (-1 for automatic)
    :return: returns trained MOG2 background model
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(bg_video_input_path, bg_video_input_filename))
    if not cap.isOpened():
        return None

    background_model = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                          detectShadows=detect_shadows)

    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            break

        # Convert to HSV
        if use_hsv:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Apply background subtractor on frame
        background_model.apply(current_frame, None, learning_rate)

    return background_model

def extract_foreground_mask(image, bg_model, learning_rate=0, figure_threshold=5000, figure_inner_threshold=115,
                            apply_opening_pre=False, apply_closing_pre=False, apply_opening_post=False,
                            apply_closing_post=False):
    """
    Extracts foreground mask from image by using a background subtraction model and thresholding. After a mask is given
    by the background model, contours larger than figure_threshold are drawn in with white. For each of these contours,
    if they have inner contours with areas larger than figure_inner_threshold then those areas are drawn in with black
    to bring back lost islands inside the contours. A series of morphological transformations before or after drawing
    the contours with parameters.

    :param image: BGR image
    :param bg_model: trained background model
    :param learning_rate: learning rate for trained background model to adapt its training
    :param figure_threshold: contour will be drawn in white if its area is larger than this threshold
    :param figure_inner_threshold: inner contour child of a contour passing figure_threshold will be drawn in black if
                                   its area is larger than this threshold
    :param apply_opening_pre: erode and then dilates (opening) to remove noise before drawing contours if True
    :param apply_closing_pre: dilate and then erodes (closing) to close small holes before drawing contours if True
                              (performed after opening if apply_opening is also True)
    :param apply_opening_post: erode and then dilates (opening) to remove noise after drawing contours if True
    :param apply_closing_post: dilate and then erodes (closing) to close small holes after drawing contours if True
                               (performed after opening if apply_opening is also True)
    :return: returns the extracted foreground mask
    """
    # Convert image to HSV
    bgr_image = deepcopy(image)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Apply background subtraction model and update it according to learning rate
    bg_model_mask = bg_model.apply(hsv_image, None, learning_rate)

    # Erode and then dilate (opening) to remove noise
    if apply_opening_pre:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg_model_mask = cv2.morphologyEx(bg_model_mask, cv2.MORPH_OPEN, kernel)

    # Dilate and then erode (closing) to close small holes
    if apply_closing_pre:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg_model_mask = cv2.morphologyEx(bg_model_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(bg_model_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty foreground that holds values ranging from 0 to 255
    foreground = np.zeros(bg_model_mask.shape, dtype=np.uint8)
    # Fill large contours and black out their inner contours
    for idx, contour in enumerate(contours):
        # Contour accepted if its area is larger than figure threshold
        if cv2.contourArea(contour) >= figure_threshold:
            # Draw contour and fill area
            cv2.drawContours(foreground, [contour], -1, 255)
            cv2.fillPoly(foreground, [contour], 255)

            # For every thresholded figure we need to fill back black areas inside it
            # Look at the first child (if it exists) as first inner contour
            inner_idx = hierarchy[0][idx][2]
            while inner_idx != -1:
                # Inner contour accepted if its area is larger than figure inner threshold
                if cv2.contourArea(contours[inner_idx], True) >= figure_inner_threshold:
                    # Draw inner contour and fill area
                    cv2.fillPoly(foreground, [contours[inner_idx]], 0)
                    cv2.drawContours(foreground, [contours[inner_idx]], -1, 255)
                # Next inner contour at the same hierarchy level as this inner contour
                inner_idx = hierarchy[0][inner_idx][0]

    # Erode and then dilate (opening) to remove noise
    if apply_opening_post:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    # Dilate and then erode (closing) to close small holes
    if apply_closing_post:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Final threshold
    foreground[foreground > 0] = 255

    return foreground


def subtract_background_from_video(bg_model, video_input_path="data/cam", video_input_filename="video.avi",
                                   frame_interval=50, stop_frame=50, result_time_visible=1000,
                                   output_frame=False, output_frame_filename="mask.jpg", learning_rate=0,
                                   figure_threshold=5000, figure_inner_threshold=115,
                                   apply_opening_pre=False, apply_closing_pre=False, apply_opening_post=False,
                                   apply_closing_post=False):
    """
    Parses frames of given video and subtracts the background to extract foreground mask.

    :param bg_model: trained background model
    :param video_input_path: training video directory path
    :param video_input_filename: training video file name (including extension)
    :param frame_interval: video frames to skip (default 50 to keep 1 frame per second of a 50 fps video)
    :param stop_frame: video frame to stop the process once reached
    :param result_time_visible: milliseconds to keep result of foreground extraction for a frame to screen, 0 to wait
                                for key press, -1 to not show result to screen
    :param output_frame: if True then outputs the first processed frame with the subtracted background and extracted
                         foreground mask in the same path as video_input_path
    :param output_frame_filename: output frame file name (including extension), used when output_frame is True
    :param learning_rate: learning rate for trained background model to adapt its training
    :param figure_threshold: contour will be drawn in white if its area is larger than this threshold
    :param figure_inner_threshold: inner contour child of a contour passing figure_threshold will be drawn in black if
                                   its area is larger than this threshold
    :param apply_opening_pre: erode and then dilates (opening) to remove noise before drawing contours if True
    :param apply_closing_pre: dilate and then erodes (closing) to close small holes before drawing contours if True
                              (performed after opening if apply_opening is also True)
    :param apply_opening_post: erode and then dilates (opening) to remove noise after drawing contours if True
    :param apply_closing_post: dilate and then erodes (closing) to close small holes after drawing contours if True
                               (performed after opening if apply_opening is also True)
    :return: returns array of extracted foreground masks for every processed video frame
    """
    # Check that video can be loaded
    cap = cv2.VideoCapture(os.path.join(video_input_path, video_input_filename))
    if not cap.isOpened():
        utils.show_warning("video_none")
        return None

    # Loop until all video frames are processed
    foregrounds = []
    frame_count = 0
    while True:
        # Read video frame
        ret_frame, current_frame = cap.read()
        # Video end
        if not ret_frame:
            break

        # Stop frame reached, stop process
        if frame_count == stop_frame:
            break

        # Check if frame will be used according to interval
        if frame_count % frame_interval == 0:
            # Extract foreground
            foreground = extract_foreground_mask(current_frame, bg_model,
                                                 learning_rate=learning_rate, figure_threshold=figure_threshold,
                                                 figure_inner_threshold=figure_inner_threshold,
                                                 apply_opening_pre=apply_opening_pre,
                                                 apply_closing_pre=apply_closing_pre,
                                                 apply_opening_post=apply_opening_post,
                                                 apply_closing_post=apply_closing_post)

            # Store extracted foreground
            foregrounds.append(foreground)

            # Show results of extraction
            if result_time_visible != -1:
                cv2.imshow("Extracted Foreground Mask", foreground)
                cv2.waitKey(result_time_visible)
                cv2.destroyAllWindows()

            # Output first processed frame
            if output_frame:
                cv2.imwrite(os.path.join(video_input_path, output_frame_filename), foreground)
                output_frame = False

        frame_count += 1

    cap.release()

    cv2.destroyAllWindows()

    return foregrounds


def plot_extracted_foreground_masks(foregrounds_knn, foregrounds_mog, foregrounds_mog2, plot_output_path="plots",
                                    plot_output_filename="background_models_mask_comparisons.png"):
    """
    Plots foreground masks which were extracted using KNN, MOG, and MOG2 background models for different cameras.

    :param foregrounds_knn: array of foreground masks extracted using KNN background model
    :param foregrounds_mog: array of foreground masks extracted using MOG background model
    :param foregrounds_mog2: array of foreground masks extracted using MOG2 background model
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    # Plot with rows equal to cameras, columns equal to models
    num_cameras = len(foregrounds_knn)
    fig, ax = plt.subplots(num_cameras, 3, figsize=(18, 5*num_cameras))
    # Reshape ax if only one camera row
    if num_cameras == 1:
        ax = ax.reshape(1, -1)

    # Plot masks for every camera in a new row
    for camera in range(num_cameras):
        # Plot KNN subtractor mask
        ax[camera, 0].set_title("Camera " + str(camera+1) + " - KNN Subtractor")
        ax[camera, 0].imshow(foregrounds_knn[camera], cmap="gray")
        ax[camera, 0].axis("off")

        # Plot MOG subtractor mask
        ax[camera, 1].set_title("Camera " + str(camera + 1) + " - MOG Subtractor")
        ax[camera, 1].imshow(foregrounds_mog[camera], cmap="gray")
        ax[camera, 1].axis("off")

        # Plot MOG2 subtractor mask
        ax[camera, 2].set_title("Camera " + str(camera + 1) + " - MOG2 Subtractor")
        ax[camera, 2].imshow(foregrounds_mog2[camera], cmap="gray")
        ax[camera, 2].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Output plot to file
    if not os.path.isdir(plot_output_path):
        os.makedirs(plot_output_path)
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()


if __name__ == '__main__':
    # Directories
    data_path = "data"
    plots_path = "plots"
    cam1_path = os.path.join(data_path, "cam1")
    cam2_path = os.path.join(data_path, "cam2")
    cam3_path = os.path.join(data_path, "cam3")
    cam4_path = os.path.join(data_path, "cam4")
    cam_paths = [cam1_path, cam2_path, cam3_path, cam4_path]

    # Background model parameters for every camera
    # figure_threshold, figure_inner_threshold,
    # apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
    cam1_bg_model_params = [5000, 115, False, False, True, True]
    cam2_bg_model_params = [5000, 115, False, False, True, True]
    cam3_bg_model_params = [5000, 175, False, True, True, True]
    cam4_bg_model_params = [5000, 115, False, False, False, True]
    cam_bg_model_params = [cam1_bg_model_params, cam2_bg_model_params, cam3_bg_model_params, cam4_bg_model_params]

    # Run background substraction for every camera
    foregrounds_knn = []
    foregrounds_mog = []
    foregrounds_mog2 = []
    for camera in range(1, len(cam_paths)+1):
        # Give frame count of video as history for background model training
        _, _, frame_count = utils.get_video_properties(cam_paths[camera-1], "background.avi")

        # Train background models
        print("Subtracting background from video frame for camera " + str(camera) + " using KNN subtractor.")
        bg_model_knn = train_KNN_background_model(cam_paths[camera-1], "background.avi", use_hsv=True,
                                                  history=frame_count, dist_threshold=3500, detect_shadows=False)

        foreground_knn = subtract_background_from_video(bg_model_knn, cam_paths[camera-1], "video.avi",
                                                        frame_interval=50, stop_frame=50, result_time_visible=-1,
                                                        output_frame=True, output_frame_filename="mask_KNN.jpg",
                                                        figure_threshold=cam_bg_model_params[camera-1][0],
                                                        figure_inner_threshold=cam_bg_model_params[camera-1][1],
                                                        apply_opening_pre=cam_bg_model_params[camera-1][2],
                                                        apply_closing_pre=cam_bg_model_params[camera-1][3],
                                                        apply_opening_post=cam_bg_model_params[camera-1][4],
                                                        apply_closing_post=cam_bg_model_params[camera-1][5])[0]

        print("Subtracting background from video frame for camera " + str(camera) + " using MOG subtractor.")
        bg_model_mog = train_MOG_background_model(cam_paths[camera-1], "background.avi", use_hsv=True,
                                                  history=frame_count, n_mixtures=50, bg_ratio=0.90, noise_sigma=0)

        foreground_mog = subtract_background_from_video(bg_model_mog, cam_paths[camera-1], "video.avi",
                                                        frame_interval=50, stop_frame=50, result_time_visible=-1,
                                                        output_frame=True, output_frame_filename="mask_MOG.jpg",
                                                        figure_threshold=cam_bg_model_params[camera-1][0],
                                                        figure_inner_threshold=cam_bg_model_params[camera-1][1],
                                                        apply_opening_pre=cam_bg_model_params[camera-1][2],
                                                        apply_closing_pre=cam_bg_model_params[camera-1][3],
                                                        apply_opening_post=cam_bg_model_params[camera-1][4],
                                                        apply_closing_post=cam_bg_model_params[camera-1][5])[0]

        print("Subtracting background from video frame for camera " + str(camera) + " using MOG2 subtractor.\n")
        bg_model_mog2 = train_MOG2_background_model(cam_paths[camera-1], "background.avi", use_hsv=True,
                                                    history=frame_count, var_threshold=650, detect_shadows=False)

        foreground_mog2 = subtract_background_from_video(bg_model_mog2, cam_paths[camera-1], "video.avi",
                                                         frame_interval=50, stop_frame=50, result_time_visible=-1,
                                                         output_frame=True, output_frame_filename="mask_MOG2.jpg",
                                                         figure_threshold=cam_bg_model_params[camera - 1][0],
                                                         figure_inner_threshold=cam_bg_model_params[camera - 1][1],
                                                         apply_opening_pre=cam_bg_model_params[camera - 1][2],
                                                         apply_closing_pre=cam_bg_model_params[camera - 1][3],
                                                         apply_opening_post=cam_bg_model_params[camera - 1][4],
                                                         apply_closing_post=cam_bg_model_params[camera - 1][5])[0]

        # Save extracted foregrounds
        foregrounds_knn.append(foreground_knn)
        foregrounds_mog.append(foreground_mog)
        foregrounds_mog2.append(foreground_mog2)

    # Plot extracted foregrounds for every camera and every background model
    plot_extracted_foreground_masks(foregrounds_knn, foregrounds_mog, foregrounds_mog2)
