import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn.model_selection import learning_curve 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def read_data(data_dir):
    X = []
    Y = []

    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'), sep='\t')
    img_list = df['file_name'].values.tolist()
    label_list = df['eye_color'].values.tolist()

    for i in range(len(img_list)):
        image = cv2.imread(os.path.join(data_dir, 'img', img_list[i]))
        image = image[220:300, 160:240]
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_LINEAR).reshape(-1)
        result = image / 255.0
        X.append(result)
        Y.append(label_list[i])

    return X, Y


if __name__ == "__main__":
    X_train, Y_train = read_data('Datasets/cartoon_set')
    print('load train data')

    train_sizes, train_loss, test_loss = learning_curve(
        RandomForestClassifier(n_estimators=100), X_train, Y_train, cv=10, scoring='neg_mean_squared_error',
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
    # 显示图例
    plt.legend(loc="best")
    plt.show()
