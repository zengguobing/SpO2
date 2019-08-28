from matplotlib import pyplot as plt
import numpy as np
from pywt._swt import swt,iswt
from PyEMD import EMD,EEMD
# from scipy.signal import argrelextrema
import logging
# from PyEMD.functions import *


# 定义需要的函数
# ---------------------------------------------------------------------------------------------
def findextema(data, type):
    """
    # 此函数用于寻找序列的极值点，包括峰值和谷值
    # 参数说明：
    # data：待寻找极值点的序列
    # type：用于确定寻找极值点的类型，'peak'为峰值，'valley'为谷值
    # 返回参数：返回极值点的索引序列
    """
    peak = []
    valley = []
    for i in range(1, len(data)-1):
        if ((data[i]-data[i-1])*(data[i+1]-data[i]) < 0) and (data[i] > 0.5*(data[i+1]+data[i-1])):
            peak.append(i)
        if ((data[i]-data[i-1])*(data[i+1]-data[i]) < 0) and (data[i] < 0.5*(data[i+1]+data[i-1])):
            valley.append(i)
    if type == 'peak':
        return peak
    if type == 'valley':
        return valley


def findpeakindex(data, lamda=0.2):
    # 初步找到峰值索引，用极值点搜索法
    # 利用高斯分布规律进行初步简单筛选

    index = findextema(data, 'peak')
    mean_peak_value = np.mean(data[index])
    std_peak_value = np.std(data[index])
    index_final = []
    for i in range(len(index)):
        if data[index[i]] > mean_peak_value-lamda*std_peak_value:
            index_final.append(index[i])
    return index_final


def findvalleyindex(data, lamda=0.2):

    #初步找到谷值索引，用极值点搜索法
    #利用高斯分布规律进行初步简单筛选
    # index = argrelextrema(data, np.less)[0]
    index = findextema(data, 'valley')
    mean_valley_value = np.mean(data[index])
    std_valley_value = np.std(data[index])
    index_final = []
    for i in range(len(index)):
        if data[index[i]] < mean_valley_value+lamda*std_valley_value:
            index_final.append(index[i])
    return index_final


def peak_sift(signal, extrema_index, fs, delta_t):

    #利用心率信息（delta_t）再次进行峰值筛选
    extrema_index_siftedout = []
    for i in range(1, len(extrema_index)):
        if signal[extrema_index[i]] - signal[extrema_index[i - 1]] > 0:
            if extrema_index[i] - extrema_index[i - 1] < 0.5 * fs * delta_t:
                extrema_index_siftedout.append(extrema_index[i - 1])
        if signal[extrema_index[i]] - signal[extrema_index[i - 1]] < 0:
            if extrema_index[i] - extrema_index[i - 1] < 0.5 * fs * delta_t:
                extrema_index_siftedout.append(extrema_index[i])
    return extrema_index_siftedout


def valley_sift(signal, extrema_index, fs, delta_t):

    #利用心率信息（delta_t）再次进行谷值筛选
    extrema_index_siftedout = []
    for i in range(1, len(extrema_index)):
        if signal[extrema_index[i]] - signal[extrema_index[i - 1]] < 0:
            if extrema_index[i] - extrema_index[i - 1] < 0.5 * fs * delta_t:
                extrema_index_siftedout.append(extrema_index[i - 1])
        if signal[extrema_index[i]] - signal[extrema_index[i - 1]] > 0:
            if extrema_index[i] - extrema_index[i - 1] < 0.5 * fs * delta_t:
                extrema_index_siftedout.append(extrema_index[i])
    return extrema_index_siftedout


def delete_zeros(data1, data2):

    #利用心率信息（delta_t）再次进行峰谷值筛选的后续步骤
    for i in range(len(data1)):
        for j in range(len(data2)):
            if data1[i] == data2[j]:
                data1[i] = 0
    temp = data1[:]
    for i in data1:
        if i == 0:
            temp.remove(i)
    return temp


def find_abnormal_points(signal, index, lamda=2):

    #两次筛选完成后再次去掉离群点
    mean_value = np.mean(signal[index])
    std = np.std(signal[index])
    abnormal_points_index = []
    for i in range(len(index)):
        if np.abs(signal[index[i]]-mean_value) > lamda*std:
            abnormal_points_index.append(index[i])
    return abnormal_points_index


if __name__ == '__main__':

    # EMD参数设定
    # logging.basicConfig(level=logging.DEBUG)

    logging.basicConfig(level=logging.INFO)

    DTYPE = np.float64
    # max_imf = 9 # 最多分解数

    ### 读入PPG信号
    # -------------------------------------------------------------

    fs = 100  # 信号采样频率（Hz）
    S_infrared = np.loadtxt('INF.txt')
    S_red = np.loadtxt('RED.txt')
    start = 3000
    signal_length = 1024
    S_infrared = S_infrared[start:signal_length + start]  # 为了方便后面利用平稳小波变换，选取2048个数据
    S_red = S_red[start:signal_length + start]
    N = S_infrared.shape[0]
    tMin, tMax = 0, N
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)

    # S_infrared = S_infrared.astype(DTYPE)
    # S_red = S_red.astype(DTYPE)
    # print("Input S_infrared.dtype: " + str(S_infrared.dtype))
    # print("Input S_red.dtype: " + str(S_red.dtype))
    # emd.FIXE_H = 5
    # emd.nbsym = 2
    # emd.spline_kind = 'cubic'
    # emd.DTYPE = DTYPE
    # ---------------------------------------------------------------------------------------

    # 进行EMD，并且画出结果
    # --------------------------------------------------------------------------------------
    # emd = EMD()
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)
    eemd.noise_width = 0.01

    imfs1 = eemd.eemd(S=S_infrared, T=None, max_imf=-1)
    imfNo1 = imfs1.shape[0]

    imfs2 = eemd.eemd(S=S_red, T=None, max_imf=-1)
    imfNo2 = imfs2.shape[0]

    c = np.floor(np.sqrt(imfNo1 + 1))
    r = np.ceil((imfNo1 + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S_infrared, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo1):
        plt.subplot(r, c, num + 2)
        plt.plot(T, imfs1[num], 'g')
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf " + str(num + 1))

    plt.show()

    c = np.floor(np.sqrt(imfNo2 + 1))
    r = np.ceil((imfNo2 + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S_red, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo2):
        plt.subplot(r, c, num + 2)
        plt.plot(T, imfs2[num], 'g')
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf " + str(num + 1))

    plt.show()
    # ------------------------------------------------------------------------------------------

    # 选取代表基线漂移伪迹的几个IMFs分量，将其置零后叠加剩余的所有IMFs分量生成处理后的信号
    # ------------------------------------------------------------------------------------------

    E_IMFs1 = imfs1
    E_IMFs2 = imfs2

    # 以极值点数判定是否为基线漂移伪迹
    imfs1_peak_count = 0
    for i in range(imfNo1):
        index = findextema(E_IMFs1[i], 'peak')
        imfs1_peak_count = len(index)
        print(imfs1_peak_count)
        if imfs1_peak_count < 0.7 * signal_length / fs:
            E_IMFs1[i] = 0

    imfs2_peak_count = 0
    for i in range(imfNo2):
        index = findextema(E_IMFs2[i], 'peak')
        imfs2_peak_count = len(index)
        print(imfs2_peak_count)
        if imfs2_peak_count < 0.7 * signal_length / fs:
            E_IMFs2[i] = 0
    # E_IMFs1[0] = 0.0
    # E_IMFs2[0] = 0.0
    #
    # E_IMFs1[imfNo1 - 3] = 0.0
    # E_IMFs2[imfNo2 - 3] = 0.0
    # E_IMFs1[imfNo1 - 2] = 0.0
    # E_IMFs2[imfNo2 - 2] = 0.0
    # E_IMFs1[imfNo1 - 1] = 0.0
    # E_IMFs2[imfNo2 - 1] = 0.0

    Sum_infrared = 0
    for i in range(imfNo1):
        Sum_infrared += E_IMFs1[i]
    Sum1_infrared = Sum_infrared + np.mean(S_infrared - Sum_infrared)

    Sum_red = 0
    for i in range(imfNo2):
        Sum_red += E_IMFs2[i]
    Sum1_red = Sum_red + np.mean(S_red - Sum_red)

    # 画出EMD去基线后的对比图
    # ---------------------------------------------------------------------------------
    plt.subplot(3, 1, 1)
    plt.plot(T, S_infrared, 'k')
    plt.title('pre-processed signal')
    plt.subplot(3, 1, 2)
    plt.plot(T, Sum1_infrared, 'r')
    plt.title('processed signal')
    plt.subplot(3, 1, 3)
    plt.plot(T, S_infrared - Sum_infrared, 'b')

    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(T, S_red, 'k')
    plt.subplot(3, 1, 2)
    plt.plot(T, Sum1_red, 'r')
    plt.subplot(3, 1, 3)
    plt.plot(T, S_red - Sum_red, 'b')

    plt.show()
    # -----------------------------------------------------------------------------------

    # 利用平稳小波变换去除高频噪音
    # -------------------------------------------------------------------------------------------
    L = 4  # 小波分解层数
    coef_red = swt(Sum1_red, 'db4', level=L)
    coef_infrared = swt(Sum1_infrared, 'db4', level=L)

    # 小波分解示意图
    plt.subplot(L + 2, 1, 1)
    plt.plot(Sum1_red)
    plt.title('red light original signal')
    for i in range(L):
        plt.subplot(L + 2, 1, i + 2)
        plt.plot(coef_red[L - i - 1][1])
    plt.subplot(L + 2, 1, L + 2)
    plt.plot(coef_red[0][0])
    plt.show()

    plt.subplot(L + 2, 1, 1)
    plt.plot(Sum1_infrared)
    plt.title('infrared original signal')
    for i in range(L):
        plt.subplot(L + 2, 1, i + 2)
        plt.plot(coef_infrared[L - i - 1][1])
    plt.subplot(L + 2, 1, L + 2)
    plt.plot(coef_infrared[0][0])
    plt.show()

    # 小波去噪，然后重构，去掉前三层小波系数
    for i in range(coef_red[0][0].shape[0]):
        coef_red[L - 1][1][i] = 0
        coef_red[L - 2][1][i] = 0
        coef_red[L - 3][1][i] = 0
        # coef_red[L-4][1][i] = 0

    for i in range(coef_infrared[0][0].shape[0]):
        coef_infrared[L - 1][1][i] = 0
        coef_infrared[L - 2][1][i] = 0
        coef_infrared[L - 3][1][i] = 0
        # coef_infrared[L-4][1][i] = 0

    Sum1_red_reconstruction = iswt(coef_red, 'db4')
    Sum1_infrared_reconstruction = iswt(coef_infrared, 'db4')

    # 将重构信号与原始信号进行对比
    plt.subplot(2, 1, 1)
    plt.plot(Sum1_red_reconstruction, color='r')
    plt.suptitle('reconstruction signal')
    plt.subplot(2, 1, 2)
    plt.plot(S_red, color='b')
    plt.suptitle('original signal')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(Sum1_infrared_reconstruction, color='r')
    plt.suptitle('reconstruction signal')
    plt.subplot(2, 1, 2)
    plt.plot(S_infrared, color='b')
    plt.suptitle('original signal')
    plt.show()
    # -----------------------------------------------------------------------------------

    # 初步寻找峰值和谷值
    # --------------------------------------------------------------------------------------------
    # valley_index_red = findpeakindex(coef_red[0][0], lamda=0.1)
    # valley_index_infrared = findpeakindex(coef_infrared[0][0], lamda=0.1)

    valley_index_red = findvalleyindex(Sum1_red_reconstruction, lamda=0)
    valley_index_infrared = findvalleyindex(Sum1_infrared_reconstruction, lamda=0.0)

    peak_index_red = findpeakindex(Sum1_red_reconstruction, lamda=0)
    peak_index_infrared = findpeakindex(Sum1_infrared_reconstruction, lamda=0)
    # ---------------------------------------------------------------------------------------------

    # 第二次峰值谷值筛选，要利用求得的心率
    # ------------------------------------------------------------------------------------------------------------------------------------
    Heart_Rate_infrared = np.argmax(abs(np.fft.fft(Sum1_infrared_reconstruction - np.mean(Sum1_infrared_reconstruction))
                                        [0:int(signal_length / 2)])) * fs / signal_length * 60
    Heart_Rate_red = np.argmax(abs(np.fft.fft(Sum1_red_reconstruction - np.mean(Sum1_red_reconstruction))
                                   [0:int(signal_length / 2)])) * fs / signal_length * 60
    Heart_Rate = (Heart_Rate_infrared + Heart_Rate_red) / 2
    delta_t = 1 / (Heart_Rate_red / 60)  # 每次心跳的间隔时间
    # 第二次谷值筛选

    valley_index_red_sifted = valley_sift(Sum1_red_reconstruction, valley_index_red, fs=fs, delta_t=delta_t)
    valley_index_infrared_sifted = valley_sift(Sum1_infrared_reconstruction, valley_index_infrared, fs=fs,
                                               delta_t=delta_t)
    valley_index_red = delete_zeros(valley_index_red, valley_index_red_sifted)
    valley_index_infrared = delete_zeros(valley_index_infrared, valley_index_infrared_sifted)
    # 第二次峰值筛选

    peak_index_red_sifted = peak_sift(Sum1_red_reconstruction, peak_index_red, fs=fs, delta_t=delta_t)
    peak_index_infrared_sifted = peak_sift(Sum1_infrared_reconstruction, peak_index_infrared, fs=fs, delta_t=delta_t)

    peak_index_red = delete_zeros(peak_index_red, peak_index_red_sifted)
    peak_index_infrared = delete_zeros(peak_index_infrared, peak_index_infrared_sifted)

    # 利用峰值和谷值频率求得心率
    HR = ((len(peak_index_red) + len(peak_index_infrared)) / 2) * 60 / (signal_length/fs)
    print("Heart Rate: ", HR)

    # 第三次峰值谷值筛选，利用高斯分布，本次筛选是为了更加准确地求得血氧浓度
    # ---------------------------------------------------------------------------------------------------------------
    # 第三次谷值筛选
    valley_index_red.remove(valley_index_red[np.argmin(Sum1_red_reconstruction[valley_index_red])])
    valley_index_infrared.remove(valley_index_infrared[np.argmin(Sum1_infrared_reconstruction[valley_index_infrared])])
    abnormal_valley_index_red = find_abnormal_points(Sum1_red_reconstruction, valley_index_red)
    abnormal_valley_index_infrared = find_abnormal_points(Sum1_infrared_reconstruction, valley_index_infrared)

    valley_index_red = delete_zeros(valley_index_red, abnormal_valley_index_red)
    valley_index_infrared = delete_zeros(valley_index_infrared, abnormal_valley_index_infrared)

    # 第三次峰值筛选
    peak_index_red.remove(peak_index_red[np.argmax(Sum1_red_reconstruction[peak_index_red])])
    peak_index_infrared.remove(peak_index_infrared[np.argmax(Sum1_infrared_reconstruction[peak_index_infrared])])
    abnormal_peak_index_red = find_abnormal_points(Sum1_red_reconstruction, peak_index_red)
    abnormal_peak_index_infrared = find_abnormal_points(Sum1_infrared_reconstruction, peak_index_infrared)

    peak_index_red = delete_zeros(peak_index_red, abnormal_peak_index_red)
    peak_index_infrared = delete_zeros(peak_index_infrared, abnormal_valley_index_infrared)

    # ------------------------------------------------------------------------------------------------------------

    # 在重建信号上标注出峰值和谷值
    # -------------------------------------------------------------------------------------------------------------
    plt.plot(Sum1_red_reconstruction, color='r')
    plt.scatter(valley_index_red, Sum1_red_reconstruction[valley_index_red], color='b')
    plt.scatter(peak_index_red, Sum1_red_reconstruction[peak_index_red], color='g')
    plt.show()

    plt.plot(Sum1_infrared_reconstruction, color='r')
    plt.scatter(valley_index_infrared, Sum1_infrared_reconstruction[valley_index_infrared], color='b')
    plt.scatter(peak_index_infrared, Sum1_infrared_reconstruction[peak_index_infrared], color='g')
    plt.show()
    # -------------------------------------------------------------------------------------------------------------

    # 求心率和血氧饱和度
    # ---------------------------------------------------------------------------------------------------------------
    # 心率

    # plt.plot(abs(np.fft.fft(Sum1_infrared_reconstruction-np.mean(Sum1_infrared_reconstruction))))
    # plt.show()

    print("心率为：", Heart_Rate_infrared, Heart_Rate_red, (Heart_Rate_red + Heart_Rate_infrared) / 2)

    # 血氧饱和度
    mean_peak_value_red = np.mean(Sum1_red_reconstruction[peak_index_red])
    mean_peak_value_infrared = np.mean(Sum1_infrared_reconstruction[peak_index_infrared])

    mean_valley_value_red = np.mean(Sum1_red_reconstruction[valley_index_red])
    mean_valley_value_infrared = np.mean(Sum1_infrared_reconstruction[valley_index_infrared])

    AC_red = mean_peak_value_red - mean_valley_value_red
    AC_infrared = mean_peak_value_infrared - mean_valley_value_infrared

    DC_red = mean_valley_value_red
    DC_infrared = mean_valley_value_infrared

    R1 = (AC_red / DC_red)/(AC_infrared / DC_infrared)
    temp1 = 104 - 17 * R1
    R2 = (AC_infrared / DC_infrared)/(AC_red / DC_red)
    temp2 = 104 - 17 * R2
    if temp1 > temp2:
        SpO2 = temp1
    else:
        SpO2 = temp2

    print("血氧饱和度为：", SpO2)






