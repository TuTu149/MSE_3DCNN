#参考：https://blog.csdn.net/weixin_42655231/article/details/103481435?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0.pc_relevant_aa&spm=1001.2101.3001.4242.1&utm_relevant_index=3
import os

import pandas as pd
import scipy.io
import scipy.io as scio
import os
import numpy as np

# path='D:/Runing_Python/mat-csv/result/SVM/SVM_result_0.2.csv'
# path='D:/Runing_Python/mat-csv/result/SVM/SVM_result_0.7.csv'
# path='D:/Runing_Python/mat-csv/result/CNN_SVM/resnet_SVM_resultt_0.7.csv'
# path='D:/Runing_Python/mat-csv/autoencoder/auto_draw/2_auto_resnet_result0-42909653.csv'
path='C:/Users/涂潮/Desktop/MSE_3DCNN.csv'



def write2txt(a):
	txtName=os.path.dirname(path)+'/'+os.path.basename(path)[:-4]+'.txt'
	f=open(txtName,'w')
	for i in a:
		s=str(i).replace('[','').replace(']','')
		s=s.replace("'",'').replace(',','')+'\n'

		f.write(s)


	f.close()
	print("write2txt finish")



def write2mat(a):
	txtName=os.path.dirname(path)+'/'+os.path.basename(path)[:-4]+'.mat'
	scio.savemat(txtName,mdict={'data':a})
	print("write2mat finish")


def opendata(path):
	df=pd.read_csv(path,header=None)
	print(type(df))
	# print(df)
	# df=df.astype({'row':'float','col':'float'}).dtypes
	df=pd.DataFrame(df,dtype=np.float)

	##dataframe对两列数据交换位置
	# df=pd.DataFrame(df,columns=['col','row'],dtype=np.float)
	df=df.round(4)
	print(df)
	# print(df)
	# for i in df:
	# 	print(type(i))
	# 	print(i)
		# i=str(i)
	list_data=df.values.tolist()
	write2txt(list_data)
	write2mat(list_data)


opendata(path)



















