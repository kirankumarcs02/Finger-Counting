import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import basic_model
import cv2

# data set has some missmatch
mapping_number = {
    "0": "3",
    "1": "0",
    "2": "3",
    "3": "3",
    "4": "1",
    "5": "3",
    "6": "4",
    "7": "3",
    "8": "2",
    "9": "5",
    " ": "",
    "": ""
}


def main():
    x_l = np.load('D:/digita_rec/finger_count/input/Sign-language-digits-dataset/X.npy')
    Y_l = np.load('D:/digita_rec/finger_count/input/Sign-language-digits-dataset/Y.npy')
    img_size = 64
    plt.subplot(1, 2, 1)
    plt.imshow(x_l[2061].reshape(img_size, img_size))
    print('SIZE X = ', x_l.shape)
    print('SIZE Y = ', Y_l)
    print('kiran', np.argmax(Y_l, axis=0))
    print('Uniq 1', x_l[np.where(Y_l == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))].shape)
    print('Uniq 2', x_l[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]].shape)

    print('x_1', Y_l[2061])
    print('Y_2', Y_l[1855])
    print('Y_2 where', np.where(Y_l == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_l[1855].reshape(img_size, img_size))
    plt.axis('off')
    plt.show()
    X = np.array(x_l)
    Y = np.array(Y_l)
    print(x_l.shape)
    print(Y_l.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    number_of_train = X_train.shape[0]
    number_of_test = X_test.shape[0]

    print('number_of_test', number_of_test)
    print('number_of_train', number_of_train)

    X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1] * X_train.shape[2])
    X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1] * X_test.shape[2])
    print("X train flatten", X_train_flatten.shape)
    print("X test flatten", X_test_flatten.shape)

    x_train = X_train_flatten.T
    x_test = X_test_flatten.T
    y_train = Y_train.T
    y_test = Y_test.T
    print("x train: ", x_train.shape)
    print("x test: ", x_test.shape)
    print("y train: ", y_train.shape)
    print("y test: ", y_test.shape)
    print('np.argmax(y_train, axis=0', np.argmax(y_train, axis=0))
    parameter = basic_model.model_nn(x_train, y_train, np.argmax(y_train, axis=0), x_test, np.argmax(y_test, axis=0), n_h=100,
                                     num_iters=1600, alpha=0.0091, print_cost=True)

    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''
        ans2 = ''
        ans3 = ''
        index = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (64, 64))
                print('cv2.resize(newImage, (28, 28))', newImage.shape)
                newImage = np.array(newImage)
                print('np.array(newImage)', newImage.shape)
                newImage = newImage.flatten()
                newImage = newImage.reshape(newImage.shape[0], 1)
                print('newImage.reshape(newImage.shape[0], 1)', newImage.shape)
                ans1 = basic_model.predict_nn(parameter, newImage)
                index = str(ans1[0])
                if index == '' or index == ' ':
                    index = '1'
        x, y, w, h = 0, 0, 300, 300
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "finger count " + mapping_number[index], (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


main()
