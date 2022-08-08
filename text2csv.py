import csv
from nturl2path import url2pathname
from random import randint
from topic import json_iterator
import json

target_ctx = []

def convert_file_into_csv (file_name = '',target_file=''):
    if target_file == '':
        target_file = file_name+'.csv'
    out = open(target_file,'w',newline='')
    csv_writer = csv.writer(out,dialect='excel')  
    with open(file_name, 'r') as f:
        for line in f.readlines():
            #  line=line.replace(',','\t')   #将每行的逗号替换成空格
            line = line[:-1]
            list = line.split('\t')
            #将字符串转为列表，从而可以按单元格写入csv
            csv_writer.writerow(list)


def get_train_data():
    source_files = [
    # './origin_dataset/biencoder-curatedtrec-train.json',
    # './origin_dataset/biencoder-nq-train.json',
    './origin_dataset/biencoder-squad1-train.json',
    './origin_dataset/biencoder-trivia-train.json',
    './origin_dataset/biencoder-webquestions-train.json'
    ]
    target_files = [source_files[i][:-5]+'.csv' for i in range(len(source_files))]

    for i in range(len(source_files)):
        total_class = 9
        source_file = source_files[i]
        target_file = target_files[i]
        target_ctxs = []
        target_ctxs.append('label,text,Target,Stance,Sentiment,SetType,OpinionTowards')
        # load json file from source_file
        json_data = json_iterator(source_file, target_file)
        for i in range(len(json_data)):
            if len(json_data[i]['positive_ctxs']) > 0 :
                ctx = json_data[i]['positive_ctxs'][0] 
                target_ctxs.append(str(randint(1,total_class))+','+ctx['text'].replace(',',' ').replace('\n',' ').replace('\t',' ')+',0,0,0,0,0')
        
        # write to file
        with open(target_file,'w+') as f:
            for ctx in target_ctxs:
                f.write(ctx+'\n')

    
if __name__ == '__main__':
    get_train_data()
