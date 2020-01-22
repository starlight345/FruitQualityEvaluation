import argparse
import os
import numpy as np
import cv2
import skimage.color
import skimage.filters
import skimage.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import adadelta
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
import svm
from PIL import Image


train = False
predict = False
fruit_type = "all"
epoch_count = 200
svm_train = False
svm_predict = False
use_lbp_feature = False
show_image = False
remove_bg = False
model_name = "model_0.0001_1000.dictionary"



def getLBPimage(img):

    # ref: https://fairyonice.github.io/implement-lbp-from%20scratch.html

    ### Step 0: Step 0: Convert an image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3
    for ih in range(0, img.shape[0] - neighboor):
        for iw in range(0, img.shape[1] - neighboor):
            ### Step 1: 3 by 3 pixel
            img = gray_image[ih:ih + neighboor, iw:iw + neighboor]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()
            ### Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector, 4)
            ### Step 3: Decimal: Convert the binary operated values to a digit.
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            imgLBP[ih + 1, iw + 1] = num
    return imgLBP


def remove_background(img):

    blur = skimage.color.rgb2gray(img)
    blur = skimage.filters.gaussian(blur, sigma=2)
    mask = blur < 0.95
    sel = np.zeros_like(img)
    sel[mask] = img[mask]

    return sel

def remove_background1(img):
    np_im = np.array(img)
    # img = mpimg.imread(img)
    # img = Image.open(img)
    # height, width, number of channels in image
    for row in range(63):  # each pixel has coordinates
        for col in range(63):
            if (np_im[row, col] <= (0, 0, 0)).all():
                if row < 70 and col < 250:
                    np_im[row, col] = [255]
                elif row < 190 and col > 300:
                    np_im[row, col] = [255]
                elif row > 150 and col < 60:
                    np_im[row, col] = [255]
                elif row > 220 and col > 190:
                    np_im[row, col] = [255]

    image_rgb = cv2.cvtColor(np_im, cv2.COLOR_BGR2RGB)

    # Rectange values: start x, start y, width, height
    rectangle = (3, 10, 320, 235)

    # Create initial mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run grabCut
    cv2.grabCut(image_rgb,  # Our image
                mask,  # The Mask
                rectangle,  # Our rectangle
                bgdModel,  # Temporary array for background
                fgdModel,  # Temporary array for background
                5,  # Number of iterations
                cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
    return image_rgb_nobg




def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# preparing training data. This includes cutting fruits out of images, storing and labeling them.
def prepare_training_data(data_folder_path, fruit_type = "all", removeBackground = remove_bg, showImage = True):
    dirs = os.listdir(data_folder_path)

    fruits = []
    labels = []
    lbp_fruits = []
    label = ""
    # for each img in each directory from dataset:
    for dir_name in dirs:

        # if directory name does not start with "fresh" or "rotten" its irrelevant data.
        if not (dir_name.startswith("fresh") or dir_name.startswith("rotten")):
            continue;

        # dir name is label of the objects.
        full_label = str(dir_name)
        if dir_name == 'freshapple' or dir_name == 'rottenapple':
            label = 'apple'
        if dir_name == 'freshorange' or dir_name == 'rottenorange':
            label = 'orange'
        if dir_name == 'freshbanana' or dir_name == 'rottenbanana':
            label = 'banana'

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
            image = cv2.resize(image, (64, 64))

            # Normalize color values to between -1 and 1
            image = (image / 128) - 1
            # remove background
            if removeBackground:
                fruit = remove_background(image)
                #fruit = np.expand_dims(fruit, axis=2)
            else:
                fruit = image
                #fruit = np.expand_dims(fruit, axis=2)
                #fruit = getLBPimage(fruit)

            if use_lbp_feature:
                lbp_fruit = getLBPimage(image)


            index += 1
            if showImage:
                if index % 10 == 0:
                    cv2.imshow(label, cv2.resize(fruit, (64, 64)))
                    cv2.waitKey(100)

            if fruit is not None:
                if label is fruit_type:
                    # add fruit to list of fruits
                    fruits.append(fruit)
                    # append label to fruit
                    labels.append(full_label)
                elif fruit_type is 'all':
                    fruits.append(fruit)
                    labels.append(label)
                    if use_lbp_feature:
                        lbp_fruits.append(lbp_fruit)



    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return fruits, labels, lbp_fruits


def mainFunction():
    total_banana, total_apple, total_orange = 0, 0, 0
    predicted_banana, predicted_apple, predicted_orange = 0, 0, 0
    predicted_fresh, predicted_rotten = 0, 0
    total_fresh, total_rotten = 0, 0
    total, correct = 0, 0
    #Preprocess Data
    if True:
        print("Preparing " + fruit_type + " data...")
        fruits, labels, lbp_fruits = prepare_training_data("dataset", fruit_type=fruit_type, removeBackground=True,
                                                           showImage=show_image)
        print("Data prepared")
        # print total fruits and labels
        print("Total " + fruit_type + ": ", len(fruits))
        print("Total labels: ", len(labels))

        if use_lbp_feature:
            fruits = np.concatenate(((np.array(fruits)), np.array(lbp_fruits).reshape(6300, 64, 64, 1)), axis=3)
            array_shape = 64 * 64 * 4
        else:
            array_shape = 64 * 64 * 3

        X_train, X_test, y_train, y_test = train_test_split(fruits, labels, test_size=0.1)
        # X_train, X_testval, y_train, y_testval = train_test_split(fruits, labels, test_size=0.2)
        # X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Encoding labels to int value
        le = preprocessing.LabelEncoder()
        le.fit(y_train)
        print("Labels")
        print(le.classes_)
        Y_train = le.transform(y_train)
        Y_test = le.transform(y_test)
        print("Labels after encoding")
        print(np.unique(Y_test))

        # Flatten data?
        X_flat_train = X_train.reshape(X_train.shape[0], array_shape)
        X_flat_test = X_test.reshape(X_test.shape[0], array_shape)

    if svm_train:
        clf = svm.MulticlassSVM(C=0.0001, tol=0.001, max_iter=1000, random_state=0, verbose=1)
        clf.fit(X_flat_train, Y_train)
        results = clf.predict(X_flat_test)

        expected = le.inverse_transform(Y_test)
        predictions = le.inverse_transform(results)
        for i in range(len(X_test)):
            if expected[i] == 'orange':
                total_orange = total_orange + 1
                if predictions[i] == 'orange':
                    predicted_orange = predicted_orange + 1
                    correct = correct + 1
            elif expected[i] == 'banana':
                total_banana = total_banana + 1
                if predictions[i] == 'banana':
                    predicted_banana = predicted_banana + 1
                    correct = correct + 1
            elif expected[i] == 'apple':
                total_apple = total_apple + 1
                if predictions[i] == 'apple':
                    predicted_apple = predicted_apple + 1
                    correct = correct + 1
            total = total + 1

    elif svm_predict:
        with open(model_name, 'rb') as config_dictionary_file:

            # Step 3
            config_dictionary = pickle.load(config_dictionary_file)
            config_dictionary.fit(X_flat_train, Y_train)
            results = config_dictionary.predict(X_flat_test)
            expected = le.inverse_transform(Y_test)
            predictions = le.inverse_transform(results)
            for i in range(len(X_test)):
                if expected[i] == 'orange':
                    total_orange = total_orange + 1
                    if predictions[i] == 'orange':
                        predicted_orange = predicted_orange + 1
                        correct = correct + 1
                elif expected[i] == 'banana':
                    total_banana = total_banana + 1
                    if predictions[i] == 'banana':
                        predicted_banana = predicted_banana + 1
                        correct = correct + 1
                elif expected[i] == 'apple':
                    total_apple = total_apple + 1
                    if predictions[i] == 'apple':
                        predicted_apple = predicted_apple + 1
                        correct = correct + 1
                total = total + 1

    if svm_train or svm_predict:
        print("Orange accuracy: " + str(predicted_orange) + "/" + str(total_orange) + " -- " +
              str(predicted_orange / total_orange * 100) + "%")
        print("Banana accuracy: " + str(predicted_banana) + "/" + str(total_banana) + " -- " +
              str(predicted_banana / total_banana * 100) + "%")
        print("Apple accuracy: " + str(predicted_apple) + "/" + str(total_apple) + " -- " +
              str(predicted_apple / total_apple * 100) + "%")
        print("Total accuracy: " + str(correct) + "/" + str(total) + " -- " + str(correct / total * 100) + "%")

    if train:

        model_cnn = Sequential()
        if lbp_fruits:
            model_cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 4)))
        else:
            model_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
        model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
        model_cnn.add(Conv2D(32, (3, 3), activation='relu'))
        model_cnn.add(Conv2D(16, (3, 3), activation='relu'))
        model_cnn.add(Dropout(0.25))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(64, activation='relu'))
        model_cnn.add(Dropout(0.5))
        model_cnn.add(Dense(3, activation='softmax'))

        opt = adadelta(lr=0.001, decay=1e-6)
        model_cnn.compile(optimizer=opt, loss=sparse_categorical_crossentropy, metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.00001)
        model_cnn.fit(X_train, Y_train,
                      batch_size=128,
                      epochs=epoch_count,
                      verbose=1,
                      validation_data=(X_test, Y_test),
                      callbacks=[reduce_lr])
        model_cnn.save(str(epoch_count) + "epoch" + fruit_type + ".h5")
        score = model_cnn.evaluate(X_test, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    if predict:
        bananas = []
        oranges = []
        apples = []
        # load model
        model = load_model(model_name)
        # summarize model.
        model.summary()
        score = model.evaluate(X_test, Y_test, verbose=0)
        predictions = model.predict_classes(X_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        expected = le.inverse_transform(Y_test)
        predictions = le.inverse_transform(predictions)

        if fruit_type is "all":
            for i in range(len(X_test)):
                if expected[i] == 'orange':
                    total_orange = total_orange + 1
                    if predictions[i] == 'orange':
                        predicted_orange = predicted_orange + 1
                        correct = correct + 1
                elif expected[i] == 'banana':
                    total_banana = total_banana + 1
                    if predictions[i] == 'banana':
                        predicted_banana = predicted_banana + 1
                        correct = correct + 1
                elif expected[i] == 'apple':
                    total_apple = total_apple + 1
                    if predictions[i] == 'apple':
                        predicted_apple = predicted_apple + 1
                        correct = correct + 1
                total = total + 1

            print("Orange accuracy: " + str(predicted_orange) + "/" + str(total_orange) + " -- " +
                  str(predicted_orange / total_orange * 100) + "%")
            print("Banana accuracy: " + str(predicted_banana) + "/" + str(total_banana) + " -- " +
                  str(predicted_banana / total_banana * 100) + "%")
            print("Apple accuracy: " + str(predicted_apple) + "/" + str(total_apple) + " -- " +
                  str(predicted_apple / total_apple * 100) + "%")
            print("Total accuracy: " + str(correct) + "/" + str(total) + " -- " + str(correct / total * 100) + "%")

        else:
            for i in range(len(X_test)):
                if expected[i] == "fresh" + fruit_type:
                    total_fresh = total_fresh + 1
                    if predictions[i] == "fresh" + fruit_type:
                        predicted_fresh = predicted_fresh + 1
                        correct = correct + 1
                elif expected[i] == "rotten" + fruit_type:
                    total_rotten = total_rotten + 1
                    if predictions[i] == "rotten" + fruit_type:
                        predicted_rotten = predicted_rotten + 1
                        correct = correct + 1
                total = total + 1

            print("Fresh " + fruit_type + " accuracy: " + str(predicted_fresh) + "/" + str(total_fresh) + " -- " +
                  str(predicted_fresh / total_fresh * 100) + "%")
            print("Rotten " + fruit_type + " accuracy: " + str(predicted_rotten) + "/" + str(total_rotten) + " -- " +
                  str(predicted_rotten / total_rotten * 100) + "%")
            print("Total accuracy: " + str(correct) + "/" + str(total) + " -- " + str(correct / total * 100) + "%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args: train_method(*), dataset(*), epoch, remove_background, show_images, model_name ')

    parser.add_argument('--method', metavar='path', type=str, required=True,
                        help='Train methods: ["svm", "cnn"]. Type one of them.')

    parser.add_argument('--predict', metavar='path', type=str, required=True, default=True,
                        help='True for predict, false for train')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Data sets: ["fruit_type", "apple_quality", "orange_quality", "banana_quality"]')

    parser.add_argument('--useLBP', type=bool, required=False,
                        help='True or False', default=False)

    parser.add_argument('--epoch', type=int, required=False,
                        help='epoch or iteration count', default=200)

    parser.add_argument('--remove_background', type=bool, required=False,
                        help='remove_background?', default=False)

    parser.add_argument('--show_images', type=bool, required=False,
                        help='show images in preprocessing? (might take a little long)', default=False)

    parser.add_argument('--model_name', required=False, default="model_0.0001_1000.dictionary",
                        help='You can type a model name to predict on')

    args = parser.parse_args()
    if args.method == "cnn":
        if args.predict:
            predict = True
        else:
            train = True
    elif args.method == "svm":
        if args.predict:
            svm_predict = True
        else:
            svm_train = True
    else:
        print("Argument error.")
    if args.dataset is "fruit_type":
        fruit_type = "all"
    elif args.dataset is "apple_quality":
        fruit_type = "apple"
    elif args.dataset is "orange_quality":
        fruit_type = "orange"
    elif args.dataset is "banana_quality":
        fruit_type = "banana"

    epoch_count = args.epoch
    use_lbp_feature = args.useLBP
    remove_bg = args.remove_background
    show_image = args.show_images
    model_name = args.model_name



    mainFunction()





