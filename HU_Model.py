from keras.utils import plot_model
import time

from datapre.MSE_3DCNN.IN.MSAM import MSAM
from datapre.MSE_3DCNN.IN.SE_block import se_block

time_start = time.time()  # 记录开始时间
import os
from datapre.write_csv_self import write_csv
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.layers import Dense, Add, Activation, Multiply
import os
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1337)  # for reproducibility
import pandas as pd
# import cv2
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Concatenate
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, cohen_kappa_score
# from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
# import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras

#
# # 读入数据
# dataset = h5py.File('C:/Users/wzl/Desktop/3Dmnist/full_dataset_vectors.h5', 'r')
#
# x_train = dataset["X_train"][:]
# x_test = dataset["X_test"][:]
#
# y_train = dataset["y_train"][:]
# y_test = dataset["y_test"][:]
try:
	from tqdm import trange
except ImportError:
	trange = range

# 读入数据
def  csv_excel(path):
	data = pd.read_csv(path,header=None)
	data=np.array(data)
	return data

def excel2m(path):
	data = pd.read_csv(path,header=None)
	data=np.array(data)
	# data = data[1:, :]
	print(data.shape)
	return data

def SVM(data):
	# data1 = e

	# 定义一个总结表
	result = []
	# all = []

	# ===========================================合并一个数组========================================

	# data1=excel2m('D:/Python运行/m/BOW/LC—BOW—SVM/b_1.xlsx')
	# table = excel2m('D:/Python运行/m/BOW/LC—BOW—SVM/UCM_row.xls')

	# data= np.array(data)
	# start = datetime.datetime.now()
	index = int(data.shape[0])
	print(index)
	X1 = []
	X2 = []
	X3 = []
	X4 = []
	X5 = []
	X6 = []
	X7 = []
	X8 = []
	X9 = []
	X10 = []
	X11 = []
	X12 = []
	X13 = []
	X14 = []
	X15 = []
	X16 = []

	dim = 7056
	result.append(dim)
	for i in range(index):

		if data[i, dim] == 1:
			X1.append(data[i])
	X1 = np.array(X1)

	for i in range(index):

		if data[i, dim] == 2:
			X2.append(data[i])
	X2 = np.array(X2)

	for i in range(index):

		if data[i, dim] == 3:
			X3.append(data[i])
	X3 = np.array(X3)

	for i in range(index):

		if data[i, dim] == 4:
			X4.append(data[i])
	X4 = np.array(X4)

	for i in range(index):

		if data[i, dim] == 5:
			X5.append(data[i])
	X5 = np.array(X5)

	for i in range(index):

		if data[i, dim] == 6:
			X6.append(data[i])
	X6 = np.array(X6)

	for i in range(index):

		if data[i, dim] == 7:
			X7.append(data[i])
	X7 = np.array(X7)

	for i in range(index):

		if data[i, dim] == 8:
			X8.append(data[i])
	X8 = np.array(X8)

	for i in range(index):

		if data[i, dim] == 9:
			X9.append(data[i])
	X9 = np.array(X9)

	for i in range(index):

		if data[i, dim] == 10:
			X10.append(data[i])
	X10 = np.array(X10)

	for i in range(index):

		if data[i, dim] == 11:
			X11.append(data[i])
	X11 = np.array(X11)

	for i in range(index):

		if data[i, dim] == 12:
			X12.append(data[i])
	X12 = np.array(X12)

	for i in range(index):

		if data[i, dim] == 13:
			X13.append(data[i])
	X13 = np.array(X13)

	for i in range(index):

		if data[i, dim] == 14:
			X14.append(data[i])
	X14 = np.array(X14)

	for i in range(index):

		if data[i, dim] == 15:
			X15.append(data[i])
	X15 = np.array(X15)

	for i in range(index):

		if data[i, dim] == 0:
			X16.append(data[i])
	X16 = np.array(X16)
	# X=np.vstack((X1,X2))
	# print len(X)
	x1, y1 = np.delete(X1, dim, axis=1), X1[:, dim]
	# print(x1.shape, y1.shape, y1)
	x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, random_state=1, train_size=0.1)

	x2, y2 = np.delete(X2, dim, axis=1), X2[:, dim]
	x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, random_state=1, train_size=0.1)

	x3, y3 = np.delete(X3, dim, axis=1), X3[:, dim]
	x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, random_state=1, train_size=0.1)

	x4, y4 = np.delete(X4, dim, axis=1), X4[:, dim]
	x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y4, random_state=1, train_size=0.1)

	x5, y5 = np.delete(X5, dim, axis=1), X5[:, dim]
	x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, random_state=1, train_size=0.1)

	x6, y6 = np.delete(X6, dim, axis=1), X6[:, dim]
	x_train6, x_test6, y_train6, y_test6 = train_test_split(x6, y6, random_state=1, train_size=0.1)

	x7, y7 = np.delete(X7, dim, axis=1), X7[:, dim]
	x_train7, x_test7, y_train7, y_test7 = train_test_split(x7, y7, random_state=1, train_size=0.1)

	x8, y8 = np.delete(X8, dim, axis=1), X8[:, dim]
	x_train8, x_test8, y_train8, y_test8 = train_test_split(x8, y8, random_state=1, train_size=0.1)

	x9, y9 = np.delete(X9, dim, axis=1), X9[:, dim]
	x_train9, x_test9, y_train9, y_test9 = train_test_split(x9, y9, random_state=1, train_size=0.1)

	x10, y10 = np.delete(X10, dim, axis=1), X10[:, dim]
	x_train10, x_test10, y_train10, y_test10 = train_test_split(x10, y10, random_state=1, train_size=0.1)

	x11, y11 = np.delete(X11, dim, axis=1), X11[:, dim]
	x_train11, x_test11, y_train11, y_test11 = train_test_split(x11, y11, random_state=1, train_size=0.1)

	x12, y12 = np.delete(X12, dim, axis=1), X12[:, dim]
	x_train12, x_test12, y_train12, y_test12 = train_test_split(x12, y12, random_state=1, train_size=0.1)

	x13, y13 = np.delete(X13, dim, axis=1), X13[:, dim]
	x_train13, x_test13, y_train13, y_test13 = train_test_split(x13, y13, random_state=1, train_size=0.1)

	x14, y14 = np.delete(X14, dim, axis=1), X14[:, dim]
	x_train14, x_test14, y_train14, y_test14 = train_test_split(x14, y14, random_state=1, train_size=0.1)

	x15, y15 = np.delete(X15, dim, axis=1), X15[:, dim]
	x_train15, x_test15, y_train15, y_test15 = train_test_split(x15, y15, random_state=1, train_size=0.1)

	x16, y16 = np.delete(X16, dim, axis=1), X16[:, dim]
	x_train16, x_test16, y_train16, y_test16 = train_test_split(x16, y16, random_state=1, train_size=0.1)

	x_train = np.vstack((x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, x_train9,
						 x_train10, x_train11, x_train12, x_train13, x_train14, x_train15, x_train16))
	y_train = np.hstack((y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8, y_train9,
						 y_train10, y_train11, y_train12, y_train13, y_train14, y_train15, y_train16))

	x_test = np.vstack((x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9, x_test10,
						x_test11, x_test12, x_test13, x_test14, x_test15, x_test16))
	y_test = np.hstack((y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8, y_test9, y_test10,
						y_test11, y_test12, y_test13, y_test14, y_test15, y_test16))
	# y_test = np.vstack((y_test1, y_test2))

	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	return x_train, y_train, x_test, y_test

# data = csv_excel('D:/pyproject/数据集/Indian Pines/27_27_3/jsindianclass27273.csv')
# data = csv_excel('D:/桌面/DATA/gf9classdate.csv')
data1=excel2m('D:/pyproject/IN数据集/win7.csv')
table=excel2m('D:/pyproject/IN数据集/label.csv')
data = np.column_stack((data1, table))
x_train, y_train, x_test, y_test=SVM(data)
x_test = data[:, 0:7056]
y_test = data[:, 7056:7057]#将y_test重新赋值

# 要使用2D的卷积，我们首先将每一张图片转化成3D的形状: width, height, channel(r/g/b).
# 要使用3D的卷积，我们首先将每一张图片转化成4D的形状: length, breadth, height, channel(r/g/b).

# np.ndarray的意思是N dimensional array
## Introduce the channel dimention in the input dataset
xtrain = np.ndarray((x_train.shape[0], 7056, 3)) # 这里的(10000, 4096, 1)是ndarray的形状，随机初始化
xtest = np.ndarray((x_test.shape[0], 7056, 3))
print('x_train.shape[0]', x_train.shape[0])  # 10000
print('x_test.shape[0]', x_test.shape[0])  # 2000

# for i in range(x_train.shape[0]):
# 	xtrain[i] = x_train[i].reshape(200, 1)
# for i in range(x_test.shape[0]):
# 	xtest[i] = x_test[i].reshape(200, 1)

## 这里有点晕，到时可以只用一个通道便好
## iterate in train and test, add the rgb dimention
# 在训练和测试中进行迭代，添加rgb尺寸
def add_rgb_dimention(array):
    scaler_map = cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, : -1]
    return array
for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimention(x_train[i])
for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimention(x_test[i])


## convert to 1 + 4D space (1st argument represents number of rows in the dataset)
xtrain = xtrain.reshape(x_train.shape[0], 7, 7, 144, 3)#(10000, 16, 16, 16, 1)
print("###",xtrain.shape)
xtest = xtest.reshape(x_test.shape[0], 7, 7, 144, 3)

y_train = keras.utils.to_categorical(y_train, 15)
# y_test = keras.utils.to_categorical(y_train, 16)
# y_test = keras.utils.to_categorical(y_test, 9)

# (10000,10)
print(y_train.shape)
print("x_train",xtrain.shape)
print("y_train",y_train.shape)
# 搭建神经网络结构
## input layer
#    假设输入数据的大小为 a1 × a2 × a3，channel数为 c，过滤器大小为f，
# 即过滤器维度为 f × f × f × c（一般不写 channel 的维度），过滤器数量为 n。
#  • 基于上述情况，三维卷积最终的输出为 ( a1 − f + 1 ) × ( a2 − f + 1 ) × ( a3 − f + 1 ) × n 。
# 该公式对于一维卷积、二维卷积仍然有效，只有去掉不相干的输入数据维度就行。
input_layer0 = Input((7, 7, 144, 3))


def FEM(input ,filters1):

	# 中间层
	X_shortcut1 = Conv3D(filters1, padding="same", kernel_size=(3, 3, 3), strides=(1, 1, 1))(input)
	X_shortcut2 = Conv3D(filters1, padding="same", kernel_size=(3, 3, 3), strides=(1, 1, 1), dilation_rate=(2, 2, 2))(X_shortcut1)

	# 第一层
	X_shortcut = Conv3D(filters1, padding="same", kernel_size=(3, 3, 3), strides=(1, 1, 1))(input)
	print("shortcut", X_shortcut.shape)

	# 第三层
	fem1 = Conv3D(filters1, kernel_size=(3, 3, 3), padding="same", activation='relu')(input)
	print("1 ", fem1.shape)
	fem2 = Conv3D(filters1, kernel_size=(3, 3, 3), padding="same", activation='relu',dilation_rate=(2, 2, 2))(fem1)
	print("2 ", fem2.shape)
	fem3 = Conv3D(filters1, kernel_size=(3, 3, 3), padding="same", activation='relu',dilation_rate=(2, 2, 2))(fem2)
	print("3 ", fem3.shape)

	X1 = Concatenate()([X_shortcut2, fem3, X_shortcut])
	print("X1", X1.shape)
	conv_layer1_1 = Conv3D(filters=16, kernel_size=(1, 1, 1), padding="same", activation='relu')(X1)
	X2 = se_block(conv_layer1_1, ratio=4)
	X2 = Activation('relu')(X2)
	return X2
# 7*7*200
conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 7),  padding="same",   activation='relu')(input_layer0)

conv_layer2=FEM(conv_layer1, 32)
X3 = Add()([conv_layer1, conv_layer2])

conv_layer3=FEM(X3, 32)
X4 = Add()([X3, conv_layer3])

X5=MSAM(X4)
# X6 = Multiply()([X4, X5])
pooling_layer = MaxPool3D(pool_size=(3, 3, 3),strides=(1, 1, 1))(X5)#(?, 1, 1, 1, 64)
print("6 ",pooling_layer.shape)
## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer1 = BatchNormalization()(pooling_layer)
flatten_layer = Flatten()(pooling_layer1)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10创建具有密集层的MLP架构
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=512, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.5)(dense_layer1)

dense_layer2 = Dense(units=256, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.5)(dense_layer2)

output_layer = Dense(units=15, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=[input_layer0], outputs=output_layer)

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.001), metrics=['acc'])#0.89
to_file = "D:/pyproject/datapre/IP/MSE_3DCNN.png"
plot_model(model, show_shapes=True, to_file=to_file)
# model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.005), metrics=['acc'])#
# model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# model.fit(x=xtrain, y=y_train, batch_size=128, epochs=50, validation_split=0.2)
plot_model(model, show_shapes=True)
model.summary()
print(xtrain.shape)#(10000, 16, 16, 16, 1)
print(y_train.shape)#(10000, 10)
#model.fit() #fit函数参数说明：https://blog.csdn.net/LuYi_WeiLin/article/details/88555813
list=[200]

for i in list:
	model.fit(xtrain, y_train, batch_size=32, epochs=int(i))
	pred = model.predict(xtest)
	pred = np.argmax(pred, axis=1)
	print(pred)
	##############
	print(pred.shape)
	print(type(pred))
	path_file_result = str('D:/pyproject/datapre/IP/')
	name = str('MSE_3DCNN')
	write_csv(pred, path_file_result, name)
	# print(y_test.shape)
	# print(type(y_test))
	accuracy = accuracy_score(y_test, pred)
	print('测试集准确率：', accuracy)
	# print pred
	# 计算混淆矩阵
	# cm_test = confusion_matrix(y_test, pred)
	# print '测试集矩阵：\n', cm_test
	# 计算每类精度
	acc_for_each_class_test = precision_score(y_test, pred, average=None)

	print('测试集每类样本精度：\n', acc_for_each_class_test)
	# 计算Kappa系数

	kappa_test = cohen_kappa_score(y_test, pred, labels=None)
	print('测试集样本Kappa：\n', kappa_test)
	# 计算平均分类精度
	average_accuracy = np.mean(acc_for_each_class_test)
	print('平均分类精度：\n', average_accuracy)
	file_txt = []
	file_txt.append(int(i))
	file_txt.append(accuracy)
	file_txt.append(kappa_test)
	file_txt.append(average_accuracy)
	file_txt = str(file_txt)
	f = open('batch_size_128.txt', 'a')
	f.write(file_txt + "\n")
	f.close()
	time_end = time.time()  # 记录结束时间
	time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
	print("运行时间",time_sum)
	run_time = round(time_sum)
	# 计算时分秒
	hour = run_time//3600
	minute = (run_time-3600*hour)//60
	second = run_time-3600*hour-60*minute
	# 输出
	print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
