import re
import xlrd
import csv
import pandas as pd
file = 'C:\\Users\\cjoder\\Desktop\\metric-error.csv'
f = pd.read_csv(file, sep=',')
df = pd.DataFrame(f)
df['label'] = '0'
size = df.shape

def new_deep(i):
    for line in open('C:\\Users\\cjoder\\Desktop\\location.txt'):
        str_list = line.split(" ")
        timestamp = re.findall(r'\d+\.?', str_list[3])[0]
        total_time = re.findall(r'\d+\.?', str_list[6])[0]
        end_time = int(total_time) + int(timestamp)
        error_tpye = str_list[0]


        if error_tpye == 'CPU':
            label = '1'
        elif error_tpye == 'MEM':
            label = '2'
        elif error_tpye == 'DISK_READ':
            label = '3'
        elif error_tpye == 'DISK_WRITE':
            label = '4'
        elif error_tpye == 'NET_IN':
            label = '5'
        else:
            label = '6'


        if i > 34908:
            return

        #找到注入起点
        while df.iloc[i, 0] < int(timestamp):
            if i > 34908:
                return
            i = i + 1
        print(i)
        #标记异常点
        if df.iloc[i, 0] >= int(timestamp):
            df.iloc[i, 11] = label
            i = i + 1
            while df.iloc[i, 0] < end_time:
                df.iloc[i, 11] = label
                i = i + 1
            #根据特定情况自动后延两个点
            df.iloc[i, 11] = label
            i = i + 1
            df.iloc[i, 11] = label
            i = i + 1


new_deep(i)
writer = pd.ExcelWriter('Mutilabel.xlsx')
df.to_excel(writer)
writer.save()
writer.close()