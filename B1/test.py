import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn.model_selection import learning_curve 
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def extract_hog_features(X):
    image_descriptors = []
    for i in range(len(X)):
        # print(i)
        fd, _ = hog(X[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                    block_norm='L2-Hys', visualize=True)
        image_descriptors.append(fd)
    return image_descriptors


def read_data(data_dir):
    X = []
    Y = []

    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'), sep='\t')
    img_list = df['file_name'].values.tolist()
    label_list = df['face_shape'].values.tolist()

    for i in range(len(img_list)):
        image = cv2.imread(os.path.join(data_dir, 'img', img_list[i]), 0)
        x_mid = image.shape[0] // 2
        y_mid = image.shape[1] // 2
        image = image[y_mid:int(y_mid * 1.5), x_mid // 2:x_mid]
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(data_dir, 'crop', img_list[i]), image)
        result = image / 255.0
        X.append(result)
        Y.append(label_list[i])

    return X, Y

if __name__ == "__main__":
    X_train, Y_train = read_data('Datasets/cartoon_set')
    X_train = extract_hog_features(X_train)
    print('load train data')

    train_sizes, train_loss, test_loss = learning_curve(
        svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr'), X_train, Y_train, cv=10, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(.1, 1.0, 5))
    
    print('train finish!')

    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
    plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
            label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    # show the graph
    plt.legend(loc="best")
    plt.show()
