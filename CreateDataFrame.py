from DDB.Data import Reader
from DDB.Service import Query
import os
import ast
from scipy import signal
import traceback
import numpy as np
import configparser
"""
算法思路：首先把各个诊断都按时间截取一致，然后得到时间长度，根据要求的输出采样率和时间算出长度，然后每个再resample成相同长度的，之后再截
取y生成，输入是时间长度和采样率，计算生成诊断等长的序列，截取的时候按步长截和x一样
时间×采样率=采样点

数据集划分思路：因为最后的结果是按炮的，而数据集是按切片的，不知道多少炮是有完整数据的，所以先不划分，先切片，切片的时候文件名标注炮号
序列和y，在生成数据集的时候去查找切片就行了，按炮号分训练集测试集和验证集
归一化：归一化参数也存储在数据库里吧，可以按需归一化
"""


def y(length, sample_rate, disruptive):
    # 自定义y,这里是破裂前30ms是1,0和1之间用15ms的三角函数过渡
    if disruptive:
        if length < 2*30*sample_rate:
            return np.ones([length])
        y_ = np.zeros([length-2*15*sample_rate])
        x = np.linspace(0, np.pi/2, 15*sample_rate)
        y_ = np.append(y_, np.sin(x))
        y_ = np.append(y_, np.ones([15*sample_rate]))
    else:
        y_ = np.zeros([length])
    return y_


class Cutter:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'DataSetConfig.ini'))
        self.tags = ast.literal_eval(config['Diagnosis']['tags'])
        self.sample_rate = int(config['Diagnosis']['sample_rate'])
        self.frame_size = int(config['Diagnosis']['frame_size'])
        self.step = int(config['Diagnosis']['step'])
        self.npy_path = config['path']['npy']
        if not os.path.exists(self.npy_path):
            os.makedirs(self.npy_path)

        ddb = Query()
        my_query = {'IsValidShot': True, 'IsDisrupt': False}
        self.shots = ddb.query(my_query)
        my_query = {'IsValidShot': True, 'IsDisrupt': True, 'CqTime': {"$gte": 0.05}, 'IpFlat': {'$gte': 110}}
        self.shots += ddb.query(my_query)

    def work(self):
        data_reader = Reader()
        ddb = Query()
        for shot in self.shots:
            if shot < 1065500 or shot > 1065599:
                continue
            print(shot)
            try:
                tags = ddb.tag(shot)
                if tags['IsDisrupt']:
                    t1 = tags['CqTime']
                else:
                    t1 = tags['RampDownTime']
                new_dig_length = int((t1 * 1000 - 50) * self.sample_rate)
                data = data_reader.read_many(shot, self.tags)
                digs = []
                for tag, (dig, time) in data.items():
                    # 诊断截取有效部分
                    digs.append(signal.resample(dig[(0.05 <= time) & (time <= t1)], new_dig_length))
                digs = np.array(digs)
                y_ = y(new_dig_length, self.sample_rate, tags['IsDisrupt'])
                index = 0
                while index + self.frame_size <= new_dig_length:
                    frame = digs[:, index: index + self.frame_size]
                    y_frame = y_[index: index + self.frame_size]
                    np.save(os.path.join(self.npy_path, 'x_{}_{}.npy'.format(shot, int(index/self.step))), frame)
                    np.save(os.path.join(self.npy_path, 'y_{}_{}.npy'.format(shot, int(index/self.step))), y_frame)
                    index += self.step
                if index + self.frame_size - new_dig_length < self.frame_size / 2:
                    frame = digs[:, new_dig_length - self.frame_size: new_dig_length]
                    y_frame = y_[new_dig_length - self.frame_size: new_dig_length]
                    np.save(os.path.join(self.npy_path, 'x_{}_{}.npy'.format(shot, int(index/self.step))), frame)
                    np.save(os.path.join(self.npy_path, 'y_{}_{}.npy'.format(shot, int(index/self.step))), y_frame)
            except Exception as e:
                print(e)
                traceback.print_exc()


if __name__ == '__main__':
    cutter = Cutter()
    cutter.work()
