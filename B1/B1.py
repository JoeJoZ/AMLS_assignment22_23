import os
import numpy
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix


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
    X_test, Y_test = read_data('Datasets/cartoon_set_test')
    X_test = extract_hog_features(X_test)
    print('load test data')
    model = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr')
    model.fit(X_train, Y_train)

    Y_predict_train = model.predict(X_train)
    acc_train = accuracy_score(Y_train, Y_predict_train)

    Y_predict_test = model.predict(X_test)
    acc_test = accuracy_score(Y_test, Y_predict_test)

    cm = confusion_matrix(Y_test, Y_predict_test)
    print(cm)
    print('Acc_train: ', acc_train)
    print('Acc_test: ', acc_test)
