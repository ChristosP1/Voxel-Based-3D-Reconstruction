import cv2
from modules.io import count_video_frames, find_file_paths, get_video_frame
from modules.utils import create_gaussian_model
import numpy as np


def sample_video_frames(video_path, step):
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

def background_subtraction(frame, mean, std, threshold):
    # The absolute difference between the frame and the mean
    diff = np.abs(frame - mean)
    # Calculate the threshold for each color channel
    threshold_value = threshold * std
    # A pixel is foreground if the difference is more than 'threshold' standard deviations
    foreground_mask = np.any(diff > threshold_value, axis=2)
    # Convert the mask to uint8 (255 for foreground, 0 for background)
    foreground_mask = (foreground_mask * 255).astype(np.uint8)

    return foreground_mask

def apply_morphological_operations(mask, kernel_size, iterations):
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Erode and then dilate (opening) to remove small white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Dilate and then erode (closing) to close small holes inside the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return mask

def apply_dilation(mask, kernel_size, iterations):
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Apply dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated_mask


if __name__ == '__main__':

    # Find the paths for intrinsic videos
    background_video_files = find_file_paths('data', 'background.avi')
    print(background_video_files)
    
    # Find the paths for extrinsic videos
    foreground_video_files = find_file_paths('data', 'video.avi')
    print(foreground_video_files)
    
    sampling_step_background = 1
    sampling_step_foreground = 25
    
    for i in range(len(background_video_files)):
        background_images, total_bg_frames = sample_video_frames(background_video_files[i], sampling_step_background)
        foreground_images, total_fg_frames = sample_video_frames(foreground_video_files[i], sampling_step_foreground)

        print("Background images:",background_images.shape[0])
        print("Foreground images", foreground_images.shape[0])
        
        mean_background, std_background = create_gaussian_model(background_images) 
        mean_foreground, std_foreground = create_gaussian_model(foreground_images) 
        
        cv2.imwrite(f'gaussian/background_mean_cam_{i+1}.jpg', mean_background)
        cv2.imwrite(f'gaussian/background_std_dev_cam_{i+1}.jpg', std_background)
        cv2.imwrite(f'gaussian/foreground_mean_cam_{i+1}.jpg', mean_foreground)
        cv2.imwrite(f'gaussian/foreground_std_dev_cam_{i+1}.jpg', std_foreground)
        
        # Get a random frame of the foreground and background video
        # random_background_frame = get_video_frame(background_video_files[i], np.random.randint(0,total_bg_frames))
        random_foreground_frame = get_video_frame(foreground_video_files[i], np.random.randint(0,total_fg_frames))
        
        threshold = 18
        # Adding artificial variation
        std_background = std_background+2.0
        foreground_mask = background_subtraction(random_foreground_frame, mean_background, std_background, threshold)
        
        foreground_mask_cleaned = apply_morphological_operations(foreground_mask, (5, 5), 1)
        # foreground_mask_cleaned = apply_morphological_operations(foreground_mask_cleaned, (10, 10), 1)

        foreground_mask_cleaned = apply_dilation(foreground_mask_cleaned, (5,5), 1)
        # foreground_mask_cleaned = apply_dilation(foreground_mask_cleaned, (1,5), 2)

        # Save or display the mask
        cv2.imwrite(f'gaussian/foreground_mask_cam_{i+1}.jpg', foreground_mask_cleaned)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    