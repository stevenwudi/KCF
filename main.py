"""
author: DI WU
stevenwudi@gmail.com
"""
import cv2
from KCF import KCFTracker
import time

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def main():
    # initialize the camera capture
    cap = cv2.VideoCapture(0)
    # initialize the tracker
    tracker = KCFTracker(feature_type='raw', sub_feature_type='gray')
    train_flag = True

    cv2.namedWindow('frame')
    cv2.setMouseCallback("frame", click_and_crop)

    rect_flag = False
    global frame
    point_loc = [500,500, 100,100]
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_clone = frame.copy()
        cv2.imshow('frame', frame)
        # Display the resulting frame
        while not rect_flag:
            clone = frame.copy()
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                frame = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        # if there are two reference points, then crop the region of interest
        # from the image and display it
            if len(refPt) == 2:
                rect_flag = True
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imshow("ROI", roi)

        if train_flag:
            print('start training the tracker...')
            start_time = time.time()
            gtRect = [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]]
            tracker.train(frame, gtRect)
            total_time = time.time() - start_time
            print("Training used time:", total_time)
            train_flag = False
        else:
            start_time = time.time()
            res = tracker.detect(frame_clone)
            total_time = time.time() - start_time
            print("Frames-per-second:", 1./ total_time)
            cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(res[0]+res[2]), int(res[1]+res[3])), (0, 255, 0), 2)
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()