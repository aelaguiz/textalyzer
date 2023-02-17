import cv2
import numpy as np
import pytesseract
import easyocr
import pprint
from image_cropping import crop_image



def extract_text(img_path, app, debug=False):
    cropped_img = crop_image(img_path, app)

    # Save the debug images to disk
    debug_path = 'debug_start_text_extraction.jpg'
    cv2.imwrite(debug_path, cropped_img)

    if app == 'imessage':
        blue_mask, gray_mask = get_imessage_masks(cropped_img, debug)
        blue_bboxes, gray_bboxes = get_contours_bboxes(cropped_img, blue_mask, gray_mask)

        print("Image size", cropped_img.shape)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(cropped_img)
        print(results)
        #data = pytesseract.image_to_data(cropped_img, output_type=pytesseract.Output.DICT)
        annotated_text = annotate_text(cropped_img, results, blue_mask, gray_mask, blue_bboxes, gray_bboxes)
        save_debug_image(cropped_img, blue_mask, gray_mask)
        return annotated_text
    else:
        text = pytesseract.image_to_string(cropped_img)
        return text


def get_imessage_masks(input_img, debug=False):
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    gray_lower = np.array([225, 225, 225], dtype=np.uint8)
    gray_upper = np.array([237, 237, 237], dtype=np.uint8)
    gray_mask = cv2.inRange(rgb_img, gray_lower, gray_upper)

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))


    if debug:
        cv2.imwrite('blue_mask.png', blue_mask)
        cv2.imwrite('gray_mask.png', gray_mask)

    return blue_mask, gray_mask


def get_contours_bboxes(cropped_img, blue_mask, gray_mask):
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_bboxes = [cv2.boundingRect(c) for c in blue_contours]
    gray_bboxes = [cv2.boundingRect(c) for c in gray_contours]

    # Draw blue bboxes in red and gray bboxes in green
    debug_img = cropped_img.copy()
    for bbox in blue_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for bbox in gray_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the debug image to disk
    cv2.imwrite('debug_bboxes.jpg', debug_img)

    return blue_bboxes, gray_bboxes


# def annotate_text(cropped_img, data, blue_mask, gray_mask, blue_bboxes, gray_bboxes):
#     annotated_lines = []
#     current_line = ''

#     prev_box = None
#     current_box = None
#     for i, text in enumerate(data['text']):
#         if text.strip():
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             bbox = (x, y, w, h)
#             color = None

#             for blue_bbox in blue_bboxes:
#                 if y > blue_bbox[1] and y + h < blue_bbox[1] + blue_bbox[3]:
#                     color = 'blue'
#                     current_box = blue_bbox
#                     break

#             for gray_bbox in gray_bboxes:
#                 if y > gray_bbox[1] and y + h < gray_bbox[1] + gray_bbox[3]:
#                     color = 'gray'
#                     current_box = gray_bbox
#                     break

#             if color:
#                 print(text, current_box, prev_box, color)
#                 if color == 'blue':
#                     prefix = 'ME: '
#                 else:
#                     prefix = 'THEM: '

#                 if current_box == prev_box:
#                     current_line += ' ' + text
#                 else:
#                     prev_box = None
#                     if current_line != '':
#                         annotated_lines.append(prefix + current_line)
#                     current_line = text

#             prev_box = current_box
#             current_box = None
                    
#     if current_line != '':
#         annotated_lines.append(prefix + current_line)

#     save_debug_image(cropped_img, blue_mask, gray_mask)
#     return '\n'.join(annotated_lines)


# def annotate_text(cropped_img, results, blue_mask, gray_mask, blue_bboxes, gray_bboxes):
#     annotated_lines = []
#     current_line = ''
#     prev_box = None
#     current_box = None
#     pp = pprint.PrettyPrinter(indent=4)
#     pp.pprint(results)

#     for i, result in enumerate(results):
#         text = result[1]
#         box = result[0]
#         # Extract x and y coordinates of each corner point
#         x1, y1 = box[0]
#         x2, y2 = box[1]
#         x3, y3 = box[2]
#         x4, y4 = box[3]
#         # Find the minimum and maximum x and y values to get the bounding box
#         x_min = min(x1, x2, x3, x4)
#         x_max = max(x1, x2, x3, x4)
#         y_min = min(y1, y2, y3, y4)
#         y_max = max(y1, y2, y3, y4)
#         bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
#         color = None

#         # print(bbox, text)
#         for blue_bbox in blue_bboxes:
#             if y_min > blue_bbox[1] and y_max < blue_bbox[1] + blue_bbox[3] \
#                     and x_min > blue_bbox[0] and x_max < blue_bbox[0] + blue_bbox[2]:
#                 color = 'blue'
#                 current_box = blue_bbox
#                 break

#         for gray_bbox in gray_bboxes:
#             if y_min > gray_bbox[1] and y_max < gray_bbox[1] + gray_bbox[3] \
#                     and x_min > gray_bbox[0] and x_max < gray_bbox[0] + gray_bbox[2]:
#                 color = 'gray'
#                 current_box = gray_bbox
#                 break

#         if color:
#             print(text, color)
#             if color == 'blue':
#                 prefix = 'ME: '
#             else:
#                 prefix = 'THEM: '

#             if current_box == prev_box:
#                 current_line += ' ' + text
#             else:
#                 prev_box = None
#                 if current_line != '':
#                     annotated_lines.append(prefix + current_line)
#                 current_line = text

#         prev_box = current_box
#         current_box = None

#     if current_line != '':
#         annotated_lines.append(prefix + current_line)

#     save_debug_image(cropped_img, blue_mask, gray_mask)
#     return '\n'.join(annotated_lines)

def annotate_text(cropped_img, results, blue_mask, gray_mask, blue_bboxes, gray_bboxes):
    annotated_text = []
    line_data = []
    for i, result in enumerate(results):
        text = result[1]
        box = result[0]
        # Extract x and y coordinates of each corner point
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        # Find the minimum and maximum x and y values to get the bounding box
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        color = None

        # print(bbox, text)
        for blue_bbox in blue_bboxes:
            if y_min > blue_bbox[1] and y_max < blue_bbox[1] + blue_bbox[3] \
                    and x_min > blue_bbox[0] and x_max < blue_bbox[0] + blue_bbox[2]:
                color = 'blue'
                break

        for gray_bbox in gray_bboxes:
            if y_min > gray_bbox[1] and y_max < gray_bbox[1] + gray_bbox[3] \
                    and x_min > gray_bbox[0] and x_max < gray_bbox[0] + gray_bbox[2]:
                color = 'gray'
                break

        if color:
            if color == 'blue':
                prefix = 'ME: '
            else:
                prefix = 'THEM: '
            line_data.append({'text': text, 'y': y_min, 'prefix': prefix})

    line_data = sorted(line_data, key=lambda x: x['y'])
    for i, data in enumerate(line_data):
        if i == 0:
            current_line = data['prefix'] + data['text']
        elif data['y'] - line_data[i - 1]['y'] < line_data[i - 1]['text'].count('\n') * 15:
            current_line += ' ' + data['prefix'] + data['text']
        else:
            annotated_text.append(current_line)
            current_line = data['prefix'] + data['text']
    annotated_text.append(current_line)

    save_debug_image(cropped_img, blue_mask, gray_mask)
    return '\n'.join(annotated_text)
    
def save_debug_image(input_img, blue_mask, gray_mask):
    # Determine the blue and gray contours in the input image
    contours_func = cv2.findContours
    contours_args = (blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = contours_func(*contours_args)
    
    contours_args = (gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray_contours, _ = contours_func(*contours_args)

    # Create a debug image with blue contours in green and gray contours in red
    debug_img = input_img.copy()
    for c in blue_contours:
        if len(c) > 0:
            cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)
    for c in gray_contours:
        if len(c) > 0:
            cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 2)

    # Save the debug image to disk
    debug_path = 'debug_contours.jpg'
    cv2.imwrite(debug_path, debug_img)
