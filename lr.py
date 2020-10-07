import numpy as np
import sys
import math


def parameter_x(word_dict, data):
    dict_len = len(word_dict)
    matrix = np.zeros(shape=(len(data), dict_len+1))
    y = []

    for i in range(len(data)):
        matrix[i][0] = 1

    for row in range(len(data)):
        label = data[row].split('\t')
        y.append(int(label[0]))
        lst = label[1:]

        feature = []
        for i in lst:
            feature_temp = i.split(':')
            feature.append(int(feature_temp[0]))

        for i in feature:
            matrix[row][i+1] = 1

    return matrix, y


def sgd(matrix, y, max_epoch, word_dict):
    dict_len = len(word_dict)
    theta = np.zeros(dict_len+1)
    learning_rate = 0.1

    for _ in range(max_epoch):
        for i in range(len(matrix)):
            num = np.exp(np.dot(theta.T, matrix[i]))
            # print(num.shape)
            den = 1 + np.exp(np.dot(theta.T, matrix[i]))
            frac = num/den
            brac = y[i] - frac
            mul = learning_rate * np.dot(matrix[i], brac)
            theta = theta + mul
    return theta


def predict(matrix, theta):
    predictions = []
    for i in range(len(matrix)):
        numerator = np.exp(np.dot(theta.T, matrix[i]))
        denominator = 1 + np.exp(np.dot(theta.T, matrix[i]))
        fraction = numerator/denominator
        prediction = round(fraction)
        predictions.append(int(prediction))
    return predictions


def get_error(y, predictions):
    Total = len(y)
    count = 0

    for i in range(Total):
        if y[i] != predictions[i]:
            count = count + 1
    error = count/Total

    return error


def neg_log_like(data, matrix, y, max_epoch, word_dict):
    dict_len = len(word_dict)
    theta = np.zeros(dict_len+1)
    learning_rate = 0.1

    for k in range(len(data)):
        for _ in range(max_epoch):
            for i in range(len(matrix)):
                plus1 = -(y[i] * np.dot(theta.T, matrix[i]))
                plus2 = math.log(1 + np.exp(np.dot(theta.T, matrix[i])))
                # total_plus = plus1 + plus2
    return total_plus


if __name__ == "__main__":
    formatted_train_input = sys.argv[1]
    formatted_valid_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    dict_open = open(dict_input, 'r')
    word_dict = {}

    for line in dict_open:
        word, number = line.strip().split(" ")
        word_dict[word] = number

    train_open = open(formatted_train_input, 'r')
    valid_open = open(formatted_valid_input, 'r')
    test_open = open(formatted_test_input, 'r')
    train_open_write = open(train_out, 'w')
    test_open_write = open(test_out, 'w')

    Data_train = []
    for i in train_open:
        Data_train.append(i.strip())
    matrix1, y1 = parameter_x(word_dict, Data_train)
    theta1 = sgd(matrix1, y1, num_epoch, word_dict)
    predictions1 = predict(matrix1, theta1)
    error1 = get_error(y1, predictions1)
    # neglog1 = neg_log_like(Data_train, matrix1, y1, num_epoch, word_dict)

    train_output = ''
    for i in predictions1:
        train_output += str(i) + '\n'
    train_open_write.write(str(train_output))

    # Data_valid = []
    # for j in valid_open:
    #     Data_valid.append(j.strip())
    #     matrix2, y2 = parameter_x(word_dict, Data_valid)
    #     theta2 = sgd(matrix2, y2, num_epoch, word_dict)
    #     predictions2 = predict(matrix2, theta2)
    #     error2 = get_error(y2, predictions2)
    #     neglog2 = neg_log_like(Data_valid, matrix2, y2, num_epoch, word_dict)

    Data_test = []
    for k in test_open:
        Data_test.append(k.strip())
    matrix3, y3 = parameter_x(word_dict, Data_test)
    # theta3 = sgd(matrix3, y3, num_epoch, word_dict)
    predictions3 = predict(matrix3, theta1)
    error3 = get_error(y3, predictions3)

    test_output = ''
    for k in predictions3:
        test_output += str(k) + '\n'
    test_open_write.write(str(test_output))

    ans1 = 'train(error): ' + str(error1)
    ans2 = 'test(error): ' + str(error3)
    print(ans1)
    print(ans2)

    metrics_out_file = open(metrics_out, 'w')
    metrics_out_file.write(ans1 + '\n')
    metrics_out_file.write(ans2)
