"""This module contains the termite tracking functionalities."""

import sys
import cv2
import detection


def track(method, video_source, template):
    """Track a single termite sample in video input.

    Args:
        method (str): tracking algorithm name.
        video_source (str): video path.
        template (str): termite template path.

    Returns:
        None.

    """
    tracker = cv2.Tracker_create(method)
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print('Could not open video.')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Could not read video file.')
        sys.exit()

    frame = cv2.resize(frame, (640, 480))
    detection.show_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    starting_point = detection.detect_sample_from_template(gray, template, 0.9)
    termite_position = (starting_point[0], starting_point[1], 20, 20)

    tracker.init(gray, termite_position)

    while True:
        ok, frame = video.read()
        frame = cv2.resize(frame, (640, 480))
        if not ok:
            break

        ok, termite_position = tracker.update(frame)

        if ok:
            p1 = (int(termite_position[0]), int(termite_position[1]))
            p2 = (int(termite_position[0] + termite_position[2]),
                  int(termite_position[1] + termite_position[3]))

            cv2.rectangle(frame, p1, p2, (0, 0, 255))

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == '__main__':
    track('KCF', '../data/termites-1.avi', '../data/sample3.jpg')
