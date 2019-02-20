# -*-coding:utf-8-*-

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn import linear_model
import cv2
import os
import numpy as np
from sklearn.model_selection import cross_val_score


def show_img(img):
    ''' 图片显示函数
    :param img: 图片文件
    :return: None
    '''
    cv2.imshow("img", img)
    cv2.waitKey(0)


def img_to_arr(img):
    '''二维图片转一维
    :param img: 图片多维数组
    :return:
    '''
    shape = img.shape
    length = shape[0] * shape[1]
    # print(length)
    # if length != 540:
    #     print('aaaaa')
    return img.reshape((1, length))[0]


def move_img(img, step):
    ''' 平移矩阵
    :param img: 多维数组，图片
    :param step: 移动不唱
    :return:多维数组
    '''
    rows, cols = img.shape
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = np.float32([[1, 0, step], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    # show_img(dst)
    return dst


def addGaussianNoise(image, percetage):
    '''高斯噪声的函数
    :param image:
    :param percetage:
    :return:
    '''
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        image[temp_x][temp_y] = 255
    # show_img(image)
    return image


def process_img(img, img_change_type='move_left'):
    '''图片处理函数
    :param img: 图片
    :param img_change_type:
    :return:
    '''
    if img_change_type == 'move_left':
        img = move_img(img, 1)
    elif img_change_type == 'move_right':
        img = move_img(img, -1)
    elif img_change_type == 'noise':
        img = addGaussianNoise(img, 0.01)
    arr = img_to_arr(np.array(img))
    return arr


def build_data(pay_type, path, img_type):
    '''构建训练数据元
    :param pay_type: 支付类型
    :param path: 文件路径
    :param img_type: reason or money or receive
    :return:
    '''
    X = []
    Y = []
    file_names = []

    for file in os.listdir(path + img_type):
        print(file)
        split_char = '.'
        if len(file.split(split_char)) >= 2:
            # print(file.split(split_char))

            img = cv2.imread(path + img_type + "/" + file, cv2.IMREAD_GRAYSCALE)

            # 支付宝普通用户选项
            if pay_type == "alipay":
                if img_type == 'money':
                    shape = (37, 80)
                elif img_type == 'reason':
                    shape = (15, 36)
                elif img_type == 'receive':
                    shape = (10, 47)

            # 支付宝商家版
            elif pay_type == "alishop":
                if img_type == 'money':
                    shape = (37, 80)
                elif img_type == 'reason':
                    shape = (15, 36)
                elif img_type == 'receive':
                    shape = (10, 47)

            # 微信支付
            elif pay_type == "wechat":
                if img_type == 'money':
                    shape = (37, 80)
                elif img_type == 'reason':
                    shape = (15, 40)
                elif img_type == 'receive':
                    shape = (15, 40)

            for i in range(2):
                img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
                # show_img(img)
                print("img shape", img.shape)

                X.append(process_img(img, 'move_left'))
                Y.append(file.split(split_char)[0])     # 训练图片3.743495102222261.png，取第一个字符为预期值
                file_names.append(file)

                X.append(process_img(img, 'move_right'))
                Y.append(file.split(split_char)[0])
                file_names.append(file)
                X.append(process_img(img, 'noise'))
                Y.append(file.split(split_char)[0])
                file_names.append(file)
    # print("X:",np.array(X))
    return np.array(X), np.array(Y), file_names


def train(pay_type="alipay", img_type='money', model='svm'):
    '''模型训练主函数
    :param pay_type: 训练类型
    :param img_type: 训练识别的板块，旨在指向训练数据的路径
    :param model:训练途径
    :return:
    '''
    X, Y, _ = build_data(pay_type, './data/' + pay_type + '/train/', img_type)

    if model == 'svm':
        clf = svm.SVC(gamma=0.001)
    elif model == 'logistic':
        clf = linear_model.LogisticRegression()
    elif model == 'boosting':
        clf = AdaBoostClassifier(n_estimators=1000)
    print("X.shape", X.shape, "Y.shape", Y.shape)


    #将训练后的规律存入到模型中
    clf.fit(X, Y)
    joblib.dump(clf, './model/' + pay_type + "_" + img_type + "_" + model + "_model_new.m")

    res = clf.score(X, Y)
    print("score", res)
    # file_names = []
    X_test, Y_test, file_names = build_data(pay_type, './data/' + pay_type + '/test/', img_type)
    res = clf.predict(X_test)
    err_num = 0

    for i in range(len(res)):
        if res[i] != Y_test[i]:
            err_num += 1
            print('error img index=', i, file_names[i], 'label=', Y_test[i], 'predict=', res[i])
    # print(res)
    # print(Y_test)
    print(model, "total num=", len(res), " error num=", err_num, ' error rate=',
          str((err_num / len(res)) * 100.0) + "%")


if __name__ == "__main__":
    '''模型训练'''
    train('alipay', 'receive', 'logistic')

# train("alipay","reason",'logistic')
# train('wechat','receive','logistic')
# train('alishop','money','logistic')
# train('alishop','receive','logistic')
