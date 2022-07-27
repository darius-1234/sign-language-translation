import cv2


# captures as many images as the user wants to input (e.g., hello will require 5 images)
def capture_images():
    # region of interest (ROI) coordinates
    top, right, bottom, left = 100, 150, 400, 450

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    img_counter = 1

    while (True):
        (t, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it

        # draw the segmented hand
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # SPACE pressed
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # resize img
        gray = cv2.resize(gray, (280, 280))
        cv2.imshow("Video Feed 1", gray)
        cv2.imshow("Video Feed", frame)
        # observe the keypress by the user
        # if the user pressed "Esc", then stop looping
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        # write img file to directory
        elif k % 256 == 32:  # space case

            img_name = "pic{}.jpg".format(img_counter)
            cv2.imwrite(img_name, gray)
            print("{} written!".format(img_name))
            img_counter += 1

    # free up memory
    camera.release()
    cv2.destroyAllWindows()

    return img_counter
