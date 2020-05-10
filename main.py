# 1,读入停用词列表
from data_utils import *

def main(input_question = '诗仙是谁'):
    stopwords = stopwordslist(filepath='./data/cn_stopwords.txt')
    # 2,加载问题list 和 答案list,以及问题向量list
    with open(file='./utils/question_answer_questionVector.pkl', mode='rb') as f:
        question_list, answer_list, questionVec_list = pickle.load(file=f)
    # 3,加载倒排表
    with open(file='./utils/invertedIdx_dict.pkl', mode='rb') as f: inverted_idx = pickle.load(file=f)
    # 4,输入一个问题,通过倒排表筛选 候选问题
    input_question = input_question
    index_list = get_questionIndex(input_q=input_question, inverted_idx=inverted_idx, stopwords=stopwords)
    print('倒排表返回的问题:', index_list)
    # 4.2 将输入问题转化为向量
    input_question_cut = sentence_process(input_question, stopwords)
    print('用户问题切割:', input_question_cut)
    sentenceVec = get_sentenceVector(questionList=input_question_cut,
                                    word2indexDict_filename = './utils/word2index_dict.pkl',
                                    index2vectorMatrix_filename = './utils/index2vector.npy')
    # 5,分别计算倒排表返回的问题的向量与用户提问的向量的余弦相似度
    if len(index_list) == 1:
        top1_answer = answer_list[index_list[0]]
    elif len(index_list) == 0:
        top1_answer = '数据库没有此类问题的答案呦'
    else:
        cos_value_list = []
        temp_dict = {}
        for i, index in enumerate(index_list):
            cos_value = cos_sim(sentenceVec, questionVec_list[index])
            cos_value_list.append(cos_value)
            temp_dict[i] = index
        top5_i = np.argsort(cos_value_list)[-1:].tolist()[::-1]
        top5_index = [temp_dict[i] for i in top5_i]
        # 返回最终的答案top5
        top5_answer = [answer_list[i] for i in top5_index]
        top1_answer = top5_answer[0]

    print('最有可能的答案:', top1_answer)
    return top1_answer

if __name__ == '__main__':
    top1_answer = main(input_question='97年还珠格格中因饰演小燕子一角而红遍亚洲的女演员是谁')