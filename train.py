import os
import numpy as np
import cv2


def remove_background(img):

    return img


# preparing training data. This includes cutting fruits out of images, storing and labeling them.
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    fruits = []
    labels = []

    # for each img in each directory from dataset:
    for dir_name in dirs:

        # if directory name does not start with "fresh" or "rotten" its irrelevant data.
        if not (dir_name.startswith("fresh") or dir_name.startswith("rotten")):
            continue;

        # dir name is label of the objects.
        label = str(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        # Matching label with related name
        subject_images_names = os.listdir(subject_dir_path)

        index = 0
        # for each image name in each folder read it, extract fruit and add to list
        for image_name in subject_images_names:

            # ignore system files
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # remove background
            fruit = remove_background(image)

            index += 1
            if index % 10 == 0:
                cv2.imshow(label, cv2.resize(fruit, (400, 500)))
                cv2.waitKey(100)

            if fruit is not None:
                # add fruit to list of fruits
                fruits.append(fruit)
                # append label to fruit
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return fruits, labels


print("Preparing data...")
fruits, labels = prepare_training_data("dataset")
print("Data prepared")

# print total fruits and labels
print("Total fruits: ", len(fruits))
print("Total labels: ", len(labels))
