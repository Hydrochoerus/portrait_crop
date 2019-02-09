"""
A script for cropping portraits.
"""

import argparse
import sys
import numpy as np
from PIL import Image
import dlib

def crop(infile, outprefix="cropped_", width_ratio=0.3, top_ratio=0.75, bottom_ratio=1.0):
    """
    Ratios are used for padding the dlib face detection to include the whole
    face in the cropped output.
    """

    outfile = outprefix + infile
    original_image = Image.open(infile)
    original_array = np.array(original_image)

    detector = dlib.get_frontal_face_detector()
    detections = detector.run(original_array)
    if not detections:
        print("NO FACES FOUND, EXITING")
        sys.exit(1)
    detection = detections[0][0]

    x_extra = int(detection.width()*width_ratio)
    top_extra = int(detection.height()*top_ratio)
    bottom_extra = int(detection.height()*bottom_ratio)

    left = detection.left()  - x_extra if detection.left() > x_extra else 0
    right = detection.right() + x_extra if detection.right() + x_extra < original_array.shape[1] \
            else original_array.shape[1]
    top = detection.top() - top_extra if detection.top() > top_extra else 0
    bottom = detection.bottom() + bottom_extra if detection.bottom() + bottom_extra < original_array.shape[0] \
             else original_array.shape[0]

    crop_array = original_array[top:bottom, left:right, :]
    cropped_image = Image.fromarray(crop_array)
    cropped_image.save(outfile)

def main():
    """
    Argument parsing for crop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('--outprefix', default="cropped_")
    parser.add_argument('--width-ratio', type=float, default=0.3)
    parser.add_argument('--top-ratio', type=float, default=0.75)
    parser.add_argument('--bottom-ratio', type=float, default=1.0)
    args = parser.parse_args()
    crop(args.infile, args.outprefix, args.width_ratio, args.top_ratio, args.bottom_ratio)

if __name__ == '__main__':
    main()
