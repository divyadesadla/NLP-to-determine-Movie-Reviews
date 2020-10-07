import numpy as np
import sys
import csv


def get_feature1(word_dict, data):
    out = []
    for row in data:
        label, review = row.split('\t')
        review_words = review.split(" ")
        # print(review_words)
        temp = []

        for j in review_words:
            if j in word_dict.keys():
                if word_dict[j] not in temp:
                    temp.append(word_dict[j])
                    print(temp)

        for k in range(len(temp)):
            temp[k] = str(temp[k]) + ':1'

        row_output = str(label) + '\t' + '\t'.join(temp)
        out.append(row_output)

    return out


def get_feature2(word_dict, data):
    out = []
    for row in data:
        label, review = row.split('\t')
        review_words = review.split(" ")
        temp = []

        for j in review_words:
            if j in word_dict.keys():
                if word_dict[j] not in temp:
                    if review_words.count(j) < 4:
                        temp.append(word_dict[j])

        for k in range(len(temp)):
            temp[k] = str(temp[k]) + ':1'

        row_output = str(label) + '\t' + '\t'.join(temp)
        out.append(row_output)

    return out


if __name__ == "__main__":
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_valid_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])

    dict_open = open(dict_input, 'r')
    word_dict = {}
    for line in dict_open:
        word, number = line.strip().split(" ")
        word_dict[word] = number

    train_open = open(train_input, 'r')
    valid_open = open(valid_input, 'r')
    test_open = open(test_input, 'r')

    Data_train = []
    for i in train_open:
        Data_train.append(i.strip())

    Data_valid = []
    for i in valid_open:
        Data_valid.append(i.strip())

    Data_test = []
    for i in test_open:
        Data_test.append(i.strip())

    if feature_flag == 1:
        train_output = get_feature1(word_dict, Data_train)
        output1 = ''
        for row in train_output:
            output1 = output1 + row + '\n'
        with open(formatted_train_out, 'w+') as file:
            file.write(output1)

        valid_output = get_feature1(word_dict, Data_valid)
        output2 = ''
        for row in valid_output:
            output2 = output2 + row + '\n'
        with open(formatted_valid_out, 'w') as file:
            file.write(output2)

        test_output = get_feature1(word_dict, Data_test)
        output3 = ''
        for row in test_output:
            output3 = output3 + row + '\n'
        with open(formatted_test_out, 'w') as file:
            file.write(output3)

    elif feature_flag == 2:
        train_output = get_feature2(word_dict, Data_train)
        output = ''
        for row in train_output:
            output = output + row + '\n'
        with open(formatted_train_out, 'w') as file:
            file.write(output)

        valid_output = get_feature2(word_dict, Data_valid)
        output2 = ''
        for row in valid_output:
            output2 = output2 + row + '\n'
        with open(formatted_valid_out, 'w') as file:
            file.write(output2)

        test_output = get_feature2(word_dict, Data_test)
        output3 = ''
        for row in test_output:
            output3 = output3 + row + '\n'
        with open(formatted_test_out, 'w') as file:
            file.write(output3)
