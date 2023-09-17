"""=============================================
   DETECTION OF ROAD LANE LINE (DETECTING VIDEO)
   ============================================="""        

#Import required modules

import cv2
import numpy as np
from matplotlib import pyplot as plt

#video masking
def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    cropped_img = cv2.bitwise_and(image, mask)
    return cropped_img


def draw_lines(image, hough_lines):
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

#processing the video

def process(img):
    height = img.shape[0]
    width = img.shape[1]
    roi_vertices = [
        (0, 650),
        (2*width/3, 2*height/3),
        (width, 1000)
    ]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

    canny = cv2.Canny(gray_img, 130, 220)

    roi_img = roi(canny, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=2)

    final_img = draw_lines(img, lines)

    return final_img

#Takes video as input

cap = cv2.VideoCapture(r"C:\Users\DELL\Downloads\road-lane-detection-master\road-lane-detection-master\Data\lane_vid2.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("lane_detection.avi", fourcc, 30.0, (frame_width, frame_height))

#showing lane detected output video

while cap.isOpened():
    ret, frame = cap.read()

    try:
        frame = process(frame)

        saved_frame.write(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        break

cap.release()
saved_frame.release()
cv2.destroyAllWindows()

