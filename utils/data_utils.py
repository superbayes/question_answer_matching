
import os,re,jieba,collections,pickle
from utils.gensim_utils import *

# 计算余弦相似度
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    try:
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
    except ValueError:
        print('')
        sim=np.zeros(300)
    return sim

#创建停用词表
def stopwordslist(filepath):
    """
    创建停用词表
    :param filepath:
    :return: list
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords=[line.strip() for line in f.readline()]
    return stopwords

# 对句子使用jieba进行分词,并去停用词
"""
1,将问题中的数字转变为"#number";
2,将所有字母转化为小写
3,过滤掉低频(频率<2)的分词
"""
def sentence_process(sentence,stopwords=[]):
    """

    :param sentence: string
    :param stopwords: stop_words列表
    :return: list ['第一台','电子','计算机','在','哪里','诞生']
    """
    # 1,句子移除标点符号
    sentence=''.join(re.findall(r"([a-zA-Z0-9\u4E00-\u9FA5,]+)", sentence.strip()))
    # 2,分词
    jieba_cut = jieba.cut(sentence)
    outstr = [x for x in jieba_cut if len(x) > 1 and x not in stopwords]

    outList=[]
    for word in outstr:
        # 3,处理数字,将数字转化为 '#number'
        # if str(word).isdigit(): word = re.sub(pattern='^[0-9]+$', repl='#number', string=str(word))
        # 4,英文统一转换为小写  5,过滤掉低频(频率<2)的分词
        word = str(word).lower()
        outList.append(word)

    return outList

# 将问答对分析解析为问题list 和 答案list
def data_process(corpusPath,stopwords):
    """
    此API主要用来读取语料问答对,并将 问题分词~去掉停用词
    :param corpusPath:语料所在路径,语料的格式如:"第一台电子计算机在哪里诞生,美国\n"
    :param stopwordsPath: 停用词的列表
    :param save2pkl_filename:将问答对保存为pkl
    :return: 两个list,一个questionList:[['第一台','电子','计算机','在','哪里','诞生'],[]], 一个answerList:['美国','答案2']
    """
    question_list = []
    answer_list = []
    with open(file=corpusPath, mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list = ''.join(re.findall(r"([a-zA-Z0-9\u4E00-\u9FA5,]+)", line.strip())).split(',')
            if len(line_list)==2:
                #对句子进行分词,并移除停用词
                question_cut = sentence_process(sentence=line_list[0],stopwords=stopwords)
                answer = line_list[1]
                question_list.append(question_cut)
                answer_list.append(answer)

    assert len(question_list) == len(answer_list)
    print('问题列表,答案列表保存成功.....')

    return question_list, answer_list

# 建立并保存一个倒排表
def save_inverted_idx(qlist=[],file='./invertedIdx_dict.pkl'):
    """
    建立一个倒排表
    :param qlist: [['第一台','电子','计算机','在','哪里','诞生'],[]]
    :return: {'女儿': [0, 291], '哪个': [0, 1, 4, 5]}
    """
    qlist_allWord_dict=[word for list in qlist for word in list] # 将[[],[]] >> []
    word_dict = collections.Counter(qlist_allWord_dict) #{中国:10,美国:9,组织部:8,......}
    inverted_idx=dict()  # 定一个一个简单的倒排表
    # 将关键词添加到倒排表中
    for key, value in word_dict.items():
        temp = []
        if value > 0 and value < 1000:
            for i,qlist_ in enumerate(qlist):
                if key in qlist_:
                    temp.append(i)
        inverted_idx[key] = temp
    # 保存
    with open(file,mode='wb') as f:
        pickle.dump(obj=inverted_idx,file=f,protocol=pickle.HIGHEST_PROTOCOL)
    print('倒排索引字典表保存成功.....')
    return inverted_idx

# TODO: 输入一个问题,返回和输入问题相关的所有语料中的问题
def get_questionIndex(input_q,inverted_idx={},stopwords=[]):
    # 从倒排表中取出相关联的索引
    outList = sentence_process(sentence=input_q,stopwords=stopwords)

    index_list = []
    for i in outList:
        if i in inverted_idx.keys(): index_list += inverted_idx[i]
    index_list = list(set(index_list))

    return index_list

# 将处理后的中文句子转化为向量,使用知乎语料预训练向量
def get_sentenceVector(questionList=[],
                       word2indexDict_filename='./word2index_dict.pkl',
                       index2vectorMatrix_filename='./index2vector.npy'):
    # 3,将问题列表转化为向量list,(使用一句话切词,求平均向量)
    # 3.1将单词转化为数字索引
    with open(file=word2indexDict_filename,mode='rb') as f:
        word2index_dict = pickle.load(file=f)
    word_index_list=[]
    for word in questionList:
        try:
            word_index_list.append(word2index_dict[word])
        except KeyError:
            # 如果字典中没有这个单词,则用0代替
            word_index_list.append(0)
    # 最终保存的列表中不包含0
    word_index_list=[word for word in word_index_list if word != 0]
    print('单句话的index:', word_index_list)

    # 3.2将索引列表转化为300维向量列表
    embeding_matrix=np.load(file=index2vectorMatrix_filename)
    sentenceVec = []
    for i in word_index_list:
        sentenceVec.append(embeding_matrix[i])
    # 3.3将一句话求平均向量,shape:(1,300)
    sentenceVec = np.mean(sentenceVec,axis=0)
    print('平均向量:',sentenceVec.shape)
    return sentenceVec

# 将处理后的中文句子转化为向量,使用知乎语料预训练向量
def get_sentenceVector_v2(questionList=[[]],
                       word2indexDict_filename='./word2index_dict.pkl',
                       index2vectorMatrix_filename='./index2vector.npy'):
    # 3,将问题列表转化为向量list,(使用一句话切词,求平均向量)
    # 3.1将单词转化为数字索引
    with open(file=word2indexDict_filename,mode='rb') as f:
        word2index_dict = pickle.load(file=f)
    # 加载词向量矩阵
    embeding_matrix = np.load(file=index2vectorMatrix_filename)

    questionVec_list =[]
    for sentenceList in questionList:
        word_index_list=[]
        for word in sentenceList:
            try:
                word_index_list.append(word2index_dict[word])
            except KeyError:
                # 如果字典中没有这个单词,则用0代替
                word_index_list.append(0)
        # 最终保存的列表中不包含0
        word_index_list=[word for word in word_index_list if word != 0]
        # 3.2将索引列表转化为300维向量列表
        sentenceVec = []
        for i in word_index_list:
            sentenceVec.append(embeding_matrix[i])
        # 3.3将一句话求平均向量,shape:(1,300)
        sentenceVec = np.mean(sentenceVec,axis=0)
        print('平均向量:',sentenceVec.shape)
        questionVec_list.append(sentenceVec)
    return questionVec_list

# 保存处理好的语料中的问题list,答案list,以及问题的向量list
def save2pkl_question_answer_questionVector():
    # 1,读入停用词列表
    stopwords = stopwordslist(filepath='../data/cn_stopwords.txt')
    # 2,加载问题list 和 答案list
    question_list, answer_list = data_process(corpusPath='../data/corpusv1/普通问答语料对.data',stopwords=stopwords)
    # 3,将问题列表转化为向量list,(使用一句话切词,求平均向量)
    questionVec_list =get_sentenceVector_v2(questionList=question_list,
                          word2indexDict_filename='./word2index_dict.pkl',
                          index2vectorMatrix_filename='./index2vector.npy')

    # 保存 经过处理的 问题list,答案list,问题向量list
    with open('./question_answer_questionVector.pkl','wb') as f:
        pickle.dump(obj=(question_list, answer_list, questionVec_list),file=f,protocol=pickle.HIGHEST_PROTOCOL)
    print('经过处理的 问题list,答案list,问题向量list 保存成功.........')

    return question_list,answer_list,questionVec_list

if __name__ == '__main__':
    # 1,读入停用词列表
    stopwords = stopwordslist(filepath='../data/cn_stopwords.txt')
    # 2,加载问题list 和 答案list,以及问题向量list
    with open(file='./question_answer_questionVector.pkl', mode='rb') as f:
        question_list, answer_list,questionVec_list=pickle.load(file=f)
    # 3,加载倒排表
    with open(file='./invertedIdx_dict.pkl',mode='rb') as f: inverted_idx = pickle.load(file=f)
    # 4,输入一个问题,通过倒排表筛选 候选问题
    input_questuon = '诗仙是谁'
    index_list = get_questionIndex(input_q=input_questuon,inverted_idx=inverted_idx,stopwords=stopwords)
    print('倒排表返回的问题:',index_list)
    # 4.2 将输入问题转化为向量
    input_questuon_cut = sentence_process(input_questuon,stopwords)
    print('用户问题切割:',input_questuon_cut)
    sentenceVec = get_sentenceVector(questionList=input_questuon_cut)
    # 5,分别计算倒排表返回的问题的向量与用户提问的向量的余弦相似度
    if len(index_list)==1:
        top1_answer = answer_list[index_list[0]]
    elif len(index_list)==0:
        top1_answer='数据库没有此类问题的答案呦'
    else:
        cos_value_list=[]
        temp_dict={}
        for i,index in enumerate(index_list):
            cos_value = cos_sim(sentenceVec, questionVec_list[index])
            cos_value_list.append(cos_value)
            temp_dict[i]=index
        top5_i = np.argsort(cos_value_list)[-1:].tolist()[::-1]
        top5_index=[temp_dict[i] for i in top5_i]
        # 返回最终的答案top5
        top5_answer=[answer_list[i] for i in top5_index]
        top1_answer=top5_answer[0]

    print('最有可能的答案:', top1_answer)
