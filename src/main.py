import os
import argparse
from detect import detect_app
from text_extraction import extract_text
from image_cropping import crop_image


def main():
    parser = argparse.ArgumentParser(description='Extract text from conversation screenshots.')
    parser.add_argument('file_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    filename = os.path.basename(args.file_path)
    if not filename.lower().endswith('.png') and not filename.lower().endswith('.jpg'):
        print("Invalid file type. Only .png and .jpg files are supported.")
        return

    print("Loading", filename)
    app = detect_app(args.file_path)
    print("App: ", app)
    text = extract_text(args.file_path, app, True)
    print("Filename: ", filename)

    print("Text: \n", text)
    print("\n")


if __name__ == '__main__':
    main()
