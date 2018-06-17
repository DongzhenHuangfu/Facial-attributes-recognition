
import csv
import numpy as np
import pickle
import glob
import matplotlib.image as mpimg
import gc


## 1. Import data

# define a function to translate the number which is type of string into type int
def trans_str2int(strlist):
    intlist = []
    for i in range(len(strlist)):
        intlist.append([])
        for j in range(len(strlist[i])):
            intlist[i].append(int(strlist[i][j]))
    return np.array(intlist, dtype = np.int16)

X = []
Y = []
facial_attributes = []

print('Reading in the labels......')
with open('./data/list_attr_celeba.csv') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        del line[0]
        Y.append(line)
    facial_attributes = Y[0]
    del Y[0]
    Y = trans_str2int(Y)
print('Finish!')

print('Reading in the pixel datas......')
images = glob.glob('./data/img/*.jpg')

for file in images:
    image = mpimg.imread(file)
    X.append(image)
print('Finish!')

print('Reading in the suggestion for spliting data......')
suggestion = []
with open('./data/list_eval_partition.csv') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        suggestion.append(line[1])
    del suggestion[0]
    suggestion = trans_str2int(suggestion)
print('Finish!')

print('The length of X is: %d' % len(X))
print('The length of Y is: %d' % len(Y))
print('The length of suggestion is: %d' % len(suggestion))

print('Spliting the data......')
x_train = []
# y_train = []
# x_valid = []
# y_valid = []
# x_test = []
# y_test = []
for i in range(len(suggestion)):
    if suggestion[i] == 0:
        x_train.append(X[i])
#         y_train.append(Y[i])
#     elif suggestion[i] == 1:
#         x_valid.append(X[i])
#         y_valid.append(Y[i])
#     else:
#         x_test.append(X[i])
#         y_test.append(Y[i])
print('Finish!')

# print('The length of training data is: %d' % len(x_train))
# print('The length of validation data is: %d' % len(x_valid))
# print('The length of testing data is: %d' % len(x_test))

del X, Y, suggestion, images
gc.collect()

## save the original data， release the memory

# print('Saving the other data')
# with open('data_original_others.pickle', 'wb') as file:
#     data_dict2 = {'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 
#                   'x_test': x_test, 'y_test': y_test, 'facial_attributes': facial_attributes}
#     del y_train, x_valid, y_valid, x_test, y_test, facial_attributes
#     gc.collect()
#     pickle.dump(data_dict2,file)   

# del data_dict2
# gc.collect()
# print('Finish!')

print('Saving the training data')
x_train0 = x_train[0:30000]
x_train1 = x_train[30000:60000]
x_train2 = x_train[60000:90000]
x_train3 = x_train[90000:120000]
x_train4 = x_train[120000:150000]
x_train5 = x_train[150000:len(x_train)]
del x_train
gc.collect()

with open('data_original_xtrain0.pickle', 'wb') as file:
    data_dict1 = {'x_train0': x_train0}
    del x_train0
    gc.collect()
    pickle.dump(data_dict1,file)

with open('data_original_xtrain1.pickle', 'wb') as file:
    data_dict1 = {'x_train1': x_train1}
    del x_train1
    gc.collect()
    pickle.dump(data_dict1,file)

with open('data_original_xtrain2.pickle', 'wb') as file:
    data_dict1 = {'x_train2': x_train2}
    del x_train2
    gc.collect()
    pickle.dump(data_dict1,file)

with open('data_original_xtrain3.pickle', 'wb') as file:
    data_dict1 = {'x_train3': x_train3}
    del x_train3
    gc.collect()
    pickle.dump(data_dict1,file)

with open('data_original_xtrain4.pickle', 'wb') as file:
    data_dict1 = {'x_train4': x_train4}
    del x_train4
    gc.collect()
    pickle.dump(data_dict1,file)

with open('data_original_xtrain5.pickle', 'wb') as file:
    data_dict1 = {'x_train5': x_train5}
    del x_train5
    gc.collect()
    pickle.dump(data_dict1,file)
    
print('Finish!')