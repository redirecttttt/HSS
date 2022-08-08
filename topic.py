
from collections import defaultdict
from concurrent.futures import process
import enum
import json
from math import dist
from random import randint
import re
import tarfile
import time
from lda import LDA
import multiprocessing as mp
import threading

tdata = []
processes = []


class topiced_data:
    def __init__(self, source_file_path, target_file_path ) -> None:
        self.mylda = LDA()
        self.target_json = []
        self.source_file = source_file_path
        self.target_file = target_file_path
        pass

    def load_file(self):
        print('Reading file %s'%(self.source_file))
        with open(self.source_file, 'r') as f:
            _data = json.load(f)
            result = []
            result.extend(_data)
            print('Preloading %s\'s data to list'%(self.source_file))
            self.data = [r for r in result if len(r['positive_ctxs'])>0]
            print('Preloading succeed: %s\n'%(self.source_file))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        json_sample = self.data[index]
        target = {}
        target['dataset']=json_sample['dataset']
        target['question']=json_sample['question']
        target['answers']=json_sample['answers']
        target['positive_ctxs']=[]
        target['negative_ctxs']=[]
        target['hard_negative_ctxs']=[]

        positive_ctxs = json_sample['positive_ctxs']
        hard_negative_ctxs = json_sample['hard_negative_ctxs']

        for ctx in positive_ctxs:
            ctx['topic'] = self.mylda.lda(ctx['text'])
            
        for ctx in hard_negative_ctxs:
            ctx['topic'] = self.mylda.lda(ctx['text'])
            
        target['positive_ctxs']=positive_ctxs
        target['hard_negative_ctxs']=hard_negative_ctxs

        self.target_json.append(target)
         
        
    def dump_to_json_file(self, num=0.0):

        process_id = num
        file = self.target_file+('-%.2f-topic.json'%num)

        print('Process %.2f: Writing target_json into file: %s'%(process_id,self.target_file))

        with open(file, 'w+') as f:
            json.dump(self.target_json, f)

        print('Process %.2f: Writing target_json file: %s finished'%(process_id,self.target_file))

def operate (data=topiced_data, index=0, version=0.0, l=0 ,r=0):
    '''
        multi process operator
    '''
    for i in range(l,r):
        data[i]
        if i%10 ==0:
            print('Process %.2f of Index %d: total is %d, %d is proceed currently.'%(version, index, r-l, i-l,))
    print('Process %.2f is finished.'%(version))
    data.dump_to_json_file(version)
   
def process_sperated ():
    '''
        add a field namely topic to each positive and hard_negative field.
    '''
    global tdata
    global processes
        
    tdata = [
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-curatedtrec-train.json', '/home/dhj/Downloads/dataset/biencoder-curatedtrec-train.topic.json'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-curatedtrec-dev.json', '/home/dhj/Downloads/dataset/biencoder-curatedtrec-dev.topic.json'),
        topiced_data('/home/dhj/Downloads/dataset/biencoder-nq-train.json', '/home/dhj/Downloads/dataset/biencoder-nq-train'),
        topiced_data('/home/dhj/Downloads/dataset/biencoder-nq-dev.json', '/home/dhj/Downloads/dataset/biencoder-nq-dev'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-squad1-train.json', '/home/dhj/Downloads/dataset/biencoder-squad1-train'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-squad1-dev.json', '/home/dhj/Downloads/dataset/biencoder-squad1-dev'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-webquestions-train.json', '/home/dhj/Downloads/dataset/biencoder-webquestions-train.topic.json'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-webquestions-dev.json', '/home/dhj/Downloads/dataset/biencoder-webquestions-dev.topic.json'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-trivia-train.json', '/home/dhj/Downloads/dataset/biencoder-trivia-train'),
        # topiced_data('/home/dhj/Downloads/dataset/biencoder-trivia-dev.json', '/home/dhj/Downloads/dataset/biencoder-trivia-dev')
    ]

    for t in tdata:
        t.load_file()

    for i,d in enumerate(tdata):
        l=0
        r=len(d)
        version = 1.0
        while(r-l>4000):
            processes.append(mp.Process(target=operate, args=(d, i, version, l, l+4000)))
            l+=4000
            version+=0.01
        processes.append(mp.Process(target=operate, args=(d, i, version, l, r)))
    
    [p.start() for p in processes]  # 开启了两个进程
    [p.join() for p in processes]   # 等待两个进程依次结束
# def and class upon this line are aim to seperated add topic to ctx.
def intergate (data_files=[], target_file=''):
    file_content = []
    file_path='./temp_sccl_able_disable/'
    for data_file in data_files:
        print('Loading %s'%data_file)
        with open(file_path+data_file, 'r') as f:
            data = json.load(f)
            file_content.extend(data)
            print('%s loaded'%data_file)
    with open(file_path+target_file, 'w+') as f:
        json.dump(file_content, f)
    print('%s loaded\n'%data_file)

def combine ():
    '''
        combime multi dataset into one
    '''
    data_files = [
        ['biencoder-curatedtrec-train.topicLDAable.json','biencoder-curatedtrec-train.topicLDAdisable.json.sccl.json'],
        ['biencoder-nq-train-topicLDAable.json','biencoder-nq-train-topicLDAdisable.json.sccl.json'],
        ['biencoder-squad1-train-topicLDAable.json','biencoder-squad1-train-topicLDAdisable.json.sccl.json'],
        ['biencoder-trivia-train-topicLDAable.json','biencoder-trivia-train-topicLDAdisable.json.sccl.json'],
        ['biencoder-webquestions-train-topicLDAable.json','biencoder-webquestions-train-topicLDAdisable.json.sccl.json'],
    ]
    target_file = [
        'biencoder-curatedtrec-train-LDA-sccl.json',
        'biencoder-nq-train-LDA-sccl.json',
        'biencoder-squad1-train-LDA-sccl.json',
        'biencoder-trivia-train-LDA-sccl.json',
        'biencoder-webquestions-train-LDA-sccl.json'
        ]
    for i,data_file in enumerate(data_files):
        intergate(data_file, target_file[i])
# def and class upon this line are aim to combine multi dataset into single

class json_iterator:
    def __init__(self, source_file, target_file) -> None:
        self.source_file = source_file
        self.target_file = target_file
        self.load_data_from_json_file()

    def load_data_from_json_file(self):
        print('Loading file %s'%self.source_file)
        with open(self.source_file, 'r') as f:
            self.json_data = json.load(f)
        print('File %s Loaded'%self.source_file)


    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        return self.json_data[index]
    
def pre_LDA ():
    target_json_files=[
        'biencoder-curatedtrec-dev.LDA.json',
        'biencoder-curatedtrec-train.LDA.json',
        'biencoder-squad1-dev-LDA.json',
        'biencoder-squad1-train-LDA.json',
        'biencoder-nq-train-LDA.json',
        'biencoder-nq-dev-LDA.json',
        'biencoder-webquestions-dev-LDA.json',
        'biencoder-webquestions-train-LDA.json',
        'biencoder-trivia-dev-LDA.json',
        'biencoder-trivia-train-LDA.json'
    ]
    source_files=[
        'biencoder-curatedtrec-dev.topic.json',
        'biencoder-curatedtrec-train.topic.json',
        'biencoder-squad1-dev-topic.json',
        'biencoder-squad1-train-topic.json',
        'biencoder-nq-train-topic.json',
        'biencoder-nq-dev-topic.json',
        'biencoder-webquestions-dev-topic.json',
        'biencoder-webquestions-train-topic.json',
        'biencoder-trivia-dev-topic.json',
        'biencoder-trivia-train-topic.json'
    ]
    for i in range(len(target_json_files)):
        target_json_file = target_json_files[i]
        target_json_content = []
        source_file = source_files[i]

        LDAable_file = source_file[:-5]+'LDAable.json'
        LDAable = []
        LDAable_content = []

        LDAdisable_file = source_file[:-5]+'LDAdisable.json'
        LDAdisable = []
        LDAdisable_content = []

        records = defaultdict(list)
        it = json_iterator(source_file,target_json_file)
        for i in range(len(it)):
            #every single add the content of json_sample into target_item
            #except negative and hard_negative
            target_json_item = {}
            target_json_item['dataset']=it[i]['dataset']
            target_json_item['question']=it[i]['question']
            target_json_item['answers']=it[i]['answers']        
            target_json_item['positive_ctxs']=it[i]['positive_ctxs']
            target_json_item['negative_ctxs']=[]
            target_json_item['hard_negative_ctxs']=[]
            temp_records = defaultdict(list)
            for j,ctx in enumerate(it[i]['positive_ctxs']):
                    # i means the i_th json_sample in the json_iterator
                    # j means the j_th positive_ctx in the positive_ctxs of the i_th json_sample
                    temp_records[ctx['topic']].append(j)
            length = 0
            topic = ''
            for k, record in enumerate(temp_records.keys()):
                if len(temp_records[record]) > length:
                    length = len(temp_records[record])
                    topic = it[i]['positive_ctxs'][temp_records[record][0]]['topic']
            target_json_item['topic']=topic   
            records[topic].append(i)    
            target_json_content.append(target_json_item)
        # get the index of LDAable and LDAdisable
        for k in records.keys():
            if len(records[k]) > 1:
                LDAable.append(records[k])
            elif len(records[k]) == 1:
                LDAdisable.extend(records[k])
        for index in LDAdisable:
            target_json_content[index]['hard_negative_ctxs'] = it[index]['hard_negative_ctxs']
            LDAable_content.append(target_json_content[index])
        for able_item in LDAable:
            avg = 10/len(able_item)+1
            for index in able_item:
                for j in able_item:
                    if j!=index:
                        cur=0
                        while(cur<avg):
                            target_json_content[index]['hard_negative_ctxs'].append(
                                target_json_content[j]['positive_ctxs'][randint(0,len(target_json_content[j]['positive_ctxs'])-1)]
                            )
                            cur+=1
                LDAdisable_content.append(target_json_content[index])
        
        with open(LDAable_file, 'w+') as f:
            json.dump(LDAable_content,f)
        with open(LDAdisable_file, 'w+') as f:
            json.dump(LDAdisable_content,f)
        

if __name__ == '__main__':
    combine()
    

