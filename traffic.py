import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras
from keras import layers

from sklearn.model_selection import train_test_split, KFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4
K_FOLDS = 5


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    #for reproducibility
    np.random.seed(1)
    tf.random.set_seed(2)

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    labels = keras.utils.to_categorical(labels)

    #using cross validation
    
    # Initialize KFold
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)

    fold_index = 0
    results = []

    # Perform K-fold cross-validation
    for train_index, val_index in kf.split(images):
        fold_index += 1
        print(f"Training on fold {fold_index}/{K_FOLDS}...")

        # Split data into training and validation sets
        x_train, x_test = images[train_index], images[val_index]
        y_train, y_test = labels[train_index], labels[val_index]

        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance on validation set
        _, accuracy = model.evaluate(x_test, y_test, verbose=2)
        results.append(accuracy)

    # Calculate average performance across all folds
    avg_accuracy = np.mean(results)
    print(f"Average validation accuracy across {K_FOLDS} folds: {avg_accuracy}")

    # without cross validation

    # # Split data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(
    #     np.array(images), np.array(labels), test_size=TEST_SIZE,
    # )

    # # Get a compiled neural network
    # model = get_model()

    # # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)

    # # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)

    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


def load_data(data_dir, save_preprocessed=True, preprocessed_filename='preprocessed_data.npz'):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    if os.path.exists(preprocessed_filename):
        print("Loading preprocessed data from file...")
        data = np.load(preprocessed_filename)
        return data['images'], data['labels']
    
    listOfImages = list()
    listOfLabels = list()

    for i in range(NUM_CATEGORIES):
        directory = os.path.join(data_dir, str(i))

        #loop on directory files
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)

            #read image as numpy array and resize it
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_HEIGHT))

            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            listOfImages.append(img)
        
            listOfLabels.append(i)

    images = np.array(listOfImages)
    labels = np.array(listOfLabels)

    if save_preprocessed:
        print("Saving preprocessed data to file...")
        np.savez(preprocessed_filename, images=images, labels=labels)
    
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.Sequential()

    # convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
    

if __name__ == "__main__":
    main()
