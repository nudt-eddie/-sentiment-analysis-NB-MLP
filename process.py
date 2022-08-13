import pandas as pd
import csv
import threading

def file_process():
    file = open(r'./process.csv',mode='a',encoding='utf-8',newline='')
    csv_write = csv.DictWriter(file,fieldnames=['description','reg'])
    csv_write.writeheader()#写入表头
    score =pd.read_csv('./data.csv')
    l = len(score)
    for i in range(0,l):
        if (score.biandao_score[i] > 0.5):
            dict_data = {'description':score.description[i],'reg':1}
            csv_write.writerow(dict_data)
        if (score.biandao_score[i] < -0.5):
            dict_data = {'description':score.description[i],'reg':-1}
            csv_write.writerow(dict_data)
        if (-0.5<=score.biandao_score[i]<=0.5):
            dict_data = {'description':score.description[i],'reg':0}
            csv_write.writerow(dict_data)

def file_process1(rate):
    file = open(r'./{}.csv'.format(rate),mode='a',encoding='utf-8',newline='')
    csv_write = csv.DictWriter(file,fieldnames=['rating','comment'])
    csv_write.writeheader()#写入表头
    rating = pd.read_csv('process.csv')
    l = len(rating)
    for i in range(0,l):
        if (rating.reg[i] == rate):
            dict_data = {'rating':rating.reg[i],'comment':rating.description[i]}
            csv_write.writerow(dict_data)
    #rating['rating']

file_process()

#多线程提取
t1 = threading.Thread(target=file_process1(1))
t2 = threading.Thread(target=file_process1(0))
t3 = threading.Thread(target=file_process1(-1))


t1.start()
t2.start()
t3.start()
print("数据提取完成！")