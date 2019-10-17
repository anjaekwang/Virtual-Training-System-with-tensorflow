import numpy as np

def aaa(arr, data,i):
    if data[i,-2] == 0:
        arr[0] += 1
    elif data[i,-2] == 1:
        arr[1] += 1
    elif data[i,-2] == 2:
        arr[2] += 1
    elif data[i,-2] == 3:
        arr[3] += 1
    elif data[i,-2] == 4:
        arr[4] += 1
    elif data[i,-2] == 5:
        arr[5] += 1
    elif data[i,-2] == 6:
        arr[6] += 1


data_fname = './error.csv'
data = np.loadtxt(data_fname, skiprows=1, delimiter=',', usecols=list(range(30*57, 30*57+2)))

error_0 = np.array([0,0,0,0,0,0,0])
error_1 = np.array([0,0,0,0,0,0,0])
error_2 = np.array([0,0,0,0,0,0,0])
error_3 = np.array([0,0,0,0,0,0,0])
error_4 = np.array([0,0,0,0,0,0,0])
error_5 = np.array([0,0,0,0,0,0,0])
error_6 = np.array([0,0,0,0,0,0,0])


for i in range(data.shape[0]):
    if data[i,-1] == 0:
        aaa(error_0, data,i)
    elif data[i,-1] == 1:
        aaa(error_1, data,i)
    elif data[i,-1] == 2:
        aaa(error_2, data,i)
    elif data[i,-1] == 3:
        aaa(error_3, data,i)
    elif data[i,-1] == 4:
        aaa(error_4, data,i)
    elif data[i,-1] == 5:
        aaa(error_5, data,i)
    elif data[i,-1] == 6:
        aaa(error_6, data,i)


print('0 lable', error_0)
print('1 lable', error_1)
print('2 lable', error_2)
print('3 lable', error_3)
print('4 lable', error_4)
print('5 lable', error_5)
print('6 lable', error_6)



