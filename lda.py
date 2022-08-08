import jieba
from gensim import corpora
import gensim
from time import *

#困惑度绘图
def draw_picture(topic_min,topic_max,words):
    x = []
    y = []
    for i in range(topic_min,topic_max+1):
       bridge = lda(i,words)
       x.append(bridge[0])
       y.append(bridge[1])

    plt.plot(x,y)
    plt.show()

class LDA:
    def __init__(self):
        self.init_stopwords()
    pass

    def word_cut(self, mytext):
        return ' '.join(jieba.lcut(mytext))

    def stop_word(self, seg=[]):
        # 将去除停用词后的切分结果写入texts中
        texts = [[word for word in document.lower().split() if word not in self.stopwords] for document in seg]
        return texts

    # 切分后加入seg中
    def get_seg (self,content=[]):
        seg = []
        for each in content:
            seg.append(self.word_cut(each))
        return seg

    # 输入需要的主题数，以及每个主题下的关键词数
    def lda(self,content='', topic=1, words=1) -> str: 
        # 将评论数据写入新的 list
        content = content.strip('\n')
        # jieba分词
        seg = self.get_seg([content])
        # 删除没用的词
        texts = self.stop_word(seg)

        # 加载语料库
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # 加载LDA模型
        ladmodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic, id2word=dictionary, passes=30)

        # 输出主题模型结果
        result = ladmodel.print_topics(num_topics=topic, num_words=words)

        l = result[0][1].find('"')
        r = result[0][1].find('"',l+1)
        return result[0][1][l+1:r] 

    def init_stopwords(self):
        with open('/home/dhj/Downloads/DPR/dpr/LDA_study/english', 'r', errors='ignore', encoding='utf-8') as stop_fi:  # 修改2 停用词路径
            lines = stop_fi.readlines()
            self.stopwords = [x.strip() for x in lines]


