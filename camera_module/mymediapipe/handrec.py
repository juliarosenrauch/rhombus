# https://github.com/google/mediapipe/blob/v0.8.4/docs/solutions/hands.md#python-solution-api
# https://google.github.io/mediapipe/solutions/hands

import cv2
import copy
import itertools
import mediapipe as mp
from collections import deque
from cvfpscalc import CvFpsCalc
import numpy as np


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    history_length = 12
    point_history = deque(maxlen=history_length)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            debug_image = copy.deepcopy(cv2.flip(image, 1))
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # interesting image stuff here
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    
                    gesture = detect_gesture(pre_processed_landmark_list)

                    point_history.append(landmark_list[8])

                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_info_text(debug_image, brect, gesture)

                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            debug_image = draw_point_history(debug_image, point_history)
            cv2.imshow('MediaPipe Hands', debug_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def detect_gesture(pre_processed_landmark_list):
    base = [pre_processed_landmark_list[0], pre_processed_landmark_list[1]]
    thumb = [pre_processed_landmark_list[8], pre_processed_landmark_list[9]]
    pointer = [pre_processed_landmark_list[16], pre_processed_landmark_list[17]]
    middle = [pre_processed_landmark_list[24], pre_processed_landmark_list[25]]
    ring = [pre_processed_landmark_list[32], pre_processed_landmark_list[33]]
    pinky = [pre_processed_landmark_list[40], pre_processed_landmark_list[41]]

    pointer_knuckle = [pre_processed_landmark_list[12], pre_processed_landmark_list[13]]
    middle_knuckle = [pre_processed_landmark_list[20], pre_processed_landmark_list[21]]
    ring_knuckle = [pre_processed_landmark_list[28], pre_processed_landmark_list[29]]
    pinky_knuckle = [pre_processed_landmark_list[36], pre_processed_landmark_list[37]]

    gesture = "None"

    # print("base position: ", base)
    # print("thumb position: ", thumb)
    # print("pointer position: ", pointer)
    # print("pointer knuckle: ", pointer_knuckle)
    # print("middle position: ", middle)
    # print("ring position: ", ring)
    # print("pinky position: ", pinky)

    # thumbs up?
    if_four_folded_fingers = folded_fingers([pointer,middle,ring,pinky], [pointer_knuckle,middle_knuckle,ring_knuckle,pinky_knuckle])
    if_thumbs_up = points_in_range([pointer,middle,ring,pinky], 0.2, None) and (base[0] - thumb[1] > 0.9) and if_four_folded_fingers
    if if_thumbs_up: gesture = "Thumbs up! :D"

    # thumbs down?
    if_thumbs_down = points_in_range([pointer,middle,ring,pinky], 0.2, None) and (thumb[1] - base[0] > 0.9) and if_four_folded_fingers
    if if_thumbs_down: gesture = "Thumbs down! D:"
    
    # point left?
    if_three_folded_fingers = folded_fingers([middle,ring,pinky], [middle_knuckle,ring_knuckle,pinky_knuckle])
    if_left_point = (base[0] - pointer[0] > 0.9) and if_three_folded_fingers
    if if_left_point: gesture = "Left"

    # point right?
    if_right_point = (pointer[0] - base[0] > 0.9) and if_three_folded_fingers
    if if_right_point: gesture = "Right"

    return gesture

def folded_fingers(tips, knuckles):
    for tip, knuckle in zip(tips, knuckles):
        if abs(knuckle[0]) <= abs(tip[0]):
            return False
    return True

def points_in_range(list_of_points, x_range, y_range):
    xs, ys = zip(*list_of_points)
    x_mean = sum(xs)/len(xs)
    y_mean = sum(ys)/len(ys)
    for point in list_of_points:
        if x_range is not None:
            if abs(point[0] - x_mean) > x_range: return False
        if y_range is not None:
            if abs(point[1] - y_mean) > y_range: return False
    return True

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(image, brect):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 1)

    return image


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    #print("temp landmark list: ", temp_landmark_list)
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    #print("temp landmark list after chain: ", temp_landmark_list)

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    #print("temp landmark list after normalize: ", temp_landmark_list)

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info_text(image, brect, hand_sign_text):
    cv2.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

if __name__ == '__main__':
    main()
