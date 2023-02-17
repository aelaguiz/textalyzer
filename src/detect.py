import os
import glob
import cv2
import numpy as np


def load_resize_image(img_path, img_size):

    # Load and resize input image
    input_img = cv2.imread(img_path)
    input_h, input_w, _ = input_img.shape
    input_scale = max(input_w, input_h) / max(img_size)
    print("input image size", input_img.shape)
    input_resized = cv2.resize(input_img, (int(input_w / input_scale), int(input_h / input_scale)))
    print("resized image size", input_resized.shape)

    return input_resized


def pad_and_center_image(img, target_size):
    h, w, _ = img.shape
    th, tw = target_size
    if h < th or w < tw:
        top_pad = (th - h) // 2
        bottom_pad = th - h - top_pad
        left_pad = (tw - w) // 2
        right_pad = tw - w - left_pad
        img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img[:th, :tw]

def load_and_resize_ref(file_path, img_size):
    ref = cv2.imread(file_path)
    ref_h, ref_w, _ = ref.shape
    ref_scale = max(ref_w, ref_h) / max(img_size)
    ref_resized = cv2.resize(ref, (int(ref_w / ref_scale), int(ref_h / ref_scale)))
    ref_padded = pad_and_center_image(ref_resized, img_size)
    return ref_padded

def load_app_refs(img_size):
     # Load and resize reference images
    app_refs = {}
    app_dirs = os.listdir("images/apps")
    for app_dir in app_dirs:
        if app_dir == '.' or app_dir == '..':
            continue
        ref_paths = glob.glob(os.path.join("images/apps", app_dir, "*.png"))
        refs = []

        for p in ref_paths:
            print("Loading",p, "debug/{0}".format(p))
            resized_ref = load_and_resize_ref(p, img_size)
            refs.append(resized_ref)
            cv2.imwrite("debug/{0}".format(p), resized_ref)

        if(refs):
            app_refs[app_dir] = refs

    return app_refs

def detect_app(img_path):
    # Define fixed image size for template matching
    img_size = (800, 400)

    input_resized = pad_and_center_image(load_resize_image(img_path, img_size), img_size)
    cv2.imwrite("debug_input_resized.jpg", input_resized)
    
    app_refs = load_app_refs(img_size)

    # Create debug image
    num_rows = len(app_refs) + 1
    num_cols = max(len(refs) for app_dir, refs in app_refs.items())
    debug_img = np.zeros((num_rows * input_resized.shape[0], (num_cols+1) * input_resized.shape[1], 3), dtype=np.uint8)

    # Add input image to debug image
    debug_img[:input_resized.shape[0], :input_resized.shape[1]] = input_resized
    debug_img = cv2.putText(debug_img, 'Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Add reference images to debug image
    for i, (app_dir, refs) in enumerate(app_refs.items()):
        debug_img = cv2.putText(debug_img, os.path.basename(app_dir), (10, (i+1)*input_resized.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        for j, ref in enumerate(refs):
            row = i+1
            col = j+1
            y0 = row*input_resized.shape[0]
            y1 = y0 + ref.shape[0]
            x0 = col*input_resized.shape[1]
            x1 = x0 + ref.shape[1]
            debug_img[y0:y1, x0:x1] = ref

    # Save debug image to disk
    cv2.imwrite("debug_app_detect.jpg", debug_img)

    match_methods = [cv2.TM_CCORR_NORMED]
    results = {}
    for app, refs in app_refs.items():
        for ref in refs:
            for meth in match_methods:
                result = cv2.matchTemplate(input_resized, ref, meth)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if app not in results or max_val > results[app][0]:
                    results[app] = (max_val, min_loc)

    # Choose the best match
    app, (max_val, min_loc) = max(results.items(), key=lambda x: x[1][0])

    # Determine which app the best match corresponds to
    app_name = app.lower()


    return app_name