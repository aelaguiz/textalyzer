import cv2
import numpy as np

def crop_image(img_path, app):
    input_img = cv2.imread(img_path)
    
    # Save the debug image to disk
    debug_path = 'debug_input.jpg'
    cv2.imwrite(debug_path, input_img)
    
    if app == 'imessage':
        # Define range of white color in RGB
        lower_white = np.array([220, 220, 220])
        upper_white = np.array([255, 255, 255])

        # Threshold the RGB image to get only white colors
        mask = cv2.inRange(input_img, lower_white, upper_white)

        # Find contours of the white image to determine bounding box
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_idx = np.argmax(areas)
        x, y, w, h = cv2.boundingRect(contours[max_idx])

        # Draw the cropping area on the original image
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop image to the bounding box
        cropped_img = input_img[y:y+h, x:x+w]

    else:
        # For other apps, simply return the original image
        cropped_img = input_img

    # Save the debug images to disk
    debug_path = 'debug_crop.jpg'
    cv2.imwrite(debug_path, input_img)
    debug_path = 'debug_mask.jpg'
    cv2.imwrite(debug_path, mask)
    
    return cropped_img
