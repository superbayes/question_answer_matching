import numpy as np
import pickle
from gensim.models import KeyedVectors

class gensim_utils():

    def __init__(self,model_path='../data/chinese_word_vectors/sgns.zhihu.bigram'):
        self.model_path=model_path
        self.cn_model=self.gensim_load_model()


    def gensim_load_model(self):
        # 使用gensim加载预训练中文分词embedding, 有可能需要等待1-2分钟
        cn_model = KeyedVectors.load_word2vec_format(self.model_path, unicode_errors="ignore")
        # vector = cn_model['山东大学']
        return cn_model


    def word2index(self,wordList=[]):
        # 将单词转化为索引
        for idx, word in enumerate(wordList):
            try:
                # fixme 将单词转换为索引（tokenize） [我,爱,中国]>>>[2,299,345]
                wordList[idx] = self.cn_model.vocab[word].index
            except KeyError:
                # 如何单词不在字典中，则输出为0
                wordList[idx] = 0
        index_list = wordList.copy()
        return index_list

    def index2word(self,indexList=[]):
        # 将索引转化为单词
        #fixme 反向tokenize (即做查找表,int2word) [2,299,345]>>>[我,爱,中国]
        word_list=[]
        for i in indexList:
            if i!=0: # 0代表 未登录词  [2,299,345]>>>[我,爱,中国]
                word_list.append(self.cn_model.index2word[i])
            else:
                word_list.append(0)
        return word_list

    def save2npy_index2vector(self,embedding_dim=300,num_words=200000,filename="./index2vector.npy"):
        #fixme 构建词向量矩阵,并将其保存位一个pkl文件
        # 1,为较小规模,只是用知乎的5000个词向量:num_words
        # 2,初始化一个自己的嵌入矩阵(后边keras中会使用) [[0.1598561,0.84162,...],[0.1598561,0.84162,...]..]
        embeding_martix=np.zeros(shape=[num_words,embedding_dim])

        for i in range(num_words):
            # cn_model.index2word[1] >>的
            # cn_model[cn_model.index2word[i]] >>向量[-3.188290e-01,...]
            embeding_martix[i,]=self.cn_model[self.cn_model.index2word[i]]
        embeding_martix=embeding_martix.astype('float32')

        # 3、检查index是否 一一对应。
        print('检查index是否 一一对应:',np.sum(self.cn_model[self.cn_model.index2word[222]]==embeding_martix[222]))

        print(embeding_martix.shape)
        #
        np.save(file=filename, arr=embeding_martix)
        return embeding_martix

    def save2pkl_word2index(self,num_words=200000,
                            index2word_file='./index2word_dict.pkl',
                            word2index_file='./word2index_dict.pkl'):
        # 将知乎中的中文单词保存为{单词:索引,我:1,...}的键值对,
        # num_words 保存前几个单词
        wordlist=[]
        for i in range(num_words):
            wordlist.append(self.cn_model.index2word[i])
        # 构建单词索引
        index2word_dict =dict(enumerate(wordlist))
        # 键值对翻转 {'a':0, 'b':1......}
        word2index_dict=dict(zip(index2word_dict.values(),index2word_dict.keys()))
        # 保存为二进制文件pkl
        with open(file=index2word_file,mode='wb') as f:
            pickle.dump(obj=index2word_dict, file=f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(file=word2index_file,mode='wb') as f:
            pickle.dump(obj=word2index_dict, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        print('保存字典成功......')




if __name__ == '__main__':
    gu = gensim_utils(model_path='../data/chinese_word_vectors/sgns.zhihu.bigram')
    gu.save2npy_index2vector(num_words=200000,filename="./index2vector.npy")
    gu.save2pkl_word2index(num_words=200000,
                           index2word_file='./index2word_dict.pkl',
                           word2index_file='./word2index_dict.pkl')




