import cv2
import os
import tensorflow as tf
import numpy as np
import csv
import re as regex
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

class GestureDetail:
    def __init__(self, gesture_key, gesture_name, output_label):
        self.gesture_key = gesture_key
        self.gesture_name = gesture_name
        self.output_label = output_label


class GestureFeature:
    def __init__(self, gesture_detail: GestureDetail, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature


def extract_feature(location, input_file, mid_frame_counter):
    frame_path = frameExtractor(location + input_file, location + "frames/", mid_frame_counter)
    if frame_path is None:
        print(f"Frame extraction failed for {input_file}")
        return None

    middle_image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if middle_image is None:
        print(f"Failed to read image from {frame_path}")
        return None

    response = HandShapeFeatureExtractor.get_instance().extract_feature(middle_image)
    return response


def decide_gesture_by_file_name(gesture_file_name):
    for gesture in gesture_details:
        if gesture.gesture_key == gesture_file_name.split('_')[0]:
            return gesture
    return None


def decide_gesture_by_name(lookup_gesture_name):
    normalized_name = lookup_gesture_name.replace(" ", "").lower()
    for gesture in gesture_details:
        if gesture.gesture_name.replace(" ", "").lower() == normalized_name:
            return gesture
    return None

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
def recognize_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)
    if video_feature is None:
        print(f"Feature extraction failed for {gesture_file_name}.")
        return GestureDetail("", "", "")

    similarities = [tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature).numpy()
                    for featureVector in featureVectorList]
    if not similarities:
        return GestureDetail("", "", "")

    best_match_index = np.argmin(similarities)
    gesture_detail = featureVectorList[best_match_index].gesture_detail
    print(f"{gesture_file_name} calculated gesture: {gesture_detail.gesture_name}")
    return gesture_detail


gesture_details = [GestureDetail("Num0", "0", "0"), GestureDetail("Num1", "1", "1"),
                   GestureDetail("Num2", "2", "2"), GestureDetail("Num3", "3", "3"),
                   GestureDetail("Num4", "4", "4"), GestureDetail("Num5", "5", "5"),
                   GestureDetail("Num6", "6", "6"), GestureDetail("Num7", "7", "7"),
                   GestureDetail("Num8", "8", "8"), GestureDetail("Num9", "9", "9"),
                   GestureDetail("FanDown", "Decrease Fan Speed", "10"),
                   GestureDetail("FanOn", "FanOn", "11"), GestureDetail("FanOff", "FanOff", "12"),
                   GestureDetail("FanUp", "Increase Fan Speed", "13"),
                   GestureDetail("LightOff", "LightOff", "14"), GestureDetail("LightOn", "LightOn", "15"),
                   GestureDetail("SetThermo", "SetThermo", "16")]

featureVectorList = []

def load_training_data(path_to_train_data):
    count = 0
    for file in os.listdir(path_to_train_data):
        if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
            gesture_detail = decide_gesture_by_file_name(file)
            if gesture_detail:
                feature = extract_feature(path_to_train_data, file, count)
                if feature is not None:
                    featureVectorList.append(GestureFeature(gesture_detail, feature))
                count += 1

def process_test_data(video_locations):
    test_count = 0
    for video_location in video_locations:
        with open('results.csv', 'w', newline='') as results_file:
            fieldnames = ['Gesture_Name', 'Output_Label']
            data_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
            # data_writer.writeheader()

            for test_file in os.listdir(video_location):
                if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
                    recognized_gesture_detail = recognize_gesture(video_location, test_file, test_count)
                    test_count += 1

                    data_writer.writerow({
                                          'Gesture_Name': recognized_gesture_detail.gesture_name,
                                          'Output_Label': recognized_gesture_detail.output_label})

if __name__ == "__main__":
    path_to_train_data = "traindata/"
    load_training_data(path_to_train_data)
    video_locations = ["test/"]
    process_test_data(video_locations)
