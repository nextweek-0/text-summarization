from flask import Flask,render_template,request
import re
import jieba
from sklearn.decomposition import PCA
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from gensim import models
import numpy as np
import pandas as pd

app = Flask(__name__)
# 结巴分词
def cut(string):
    return list(jieba.cut(string))
# 拆分成列表
def sentenceslist(article):
    setences = re.sub('([。！？\?])([^”’])', r"\1 \2", article)  # 单字符断句符
    setences = re.sub('(\.{6})([^”’])', r"\1 \2", setences)  # 英文省略号
    setences = re.sub('(\…{2})([^”’])', r"\1  \2", setences)  # 中文省略号
    setences = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1 \2', setences)
    sentences_list = setences.split(' ')
    sentence_list_list = []
    for sentence in sentences_list:
        sentence = cut(sentence)
        sentence_list_list.append(sentence)
    return sentences_list,sentence_list_list
# PCA与句子向量
def sentence_to_vec(setencelist, a = 0.0001):
    model = models.Word2Vec.load("./model/wiki_corpus.model")
    sentence_set = []
    j = 0
    for sentence in setencelist:
        vs = np.zeros((250,))  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in model:
                w_v = model[word]
                p_w = model.wv.vocab[word].count
                a_value = a / (a + p_w/459358208)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, w_v))
            else:
                continue
        vs = np.divide(vs, sentence_length)
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
#         print(vs)
    # calculate PCA of this sentence set
#     sentence_set = np.mat
# rix(sentence_set).astype(np.float)
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < 250:
        for i in range(250 - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    all_vs = np.zeros((250,))
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        fin_vs = np.subtract(vs, sub)
        sentence_vecs.append(fin_vs)
        all_vs = np.add(all_vs,fin_vs)
    sentence_vecs.append(all_vs)
    return sentence_vecs
# 生成摘要
def sentences_sort(sentences_list,sentence_vecs):
    score_dict = {}
    for i in range(len(sentence_vecs)-1):
        vs  = sentence_vecs[i]
        all_vs = sentence_vecs[-1]
        a = []
        a.append(vs)
        a.append(all_vs)
        sim = cosine_similarity(a)
        score = sim[1,0]
        score_dict[i] = score
    score_list_sort = sorted(score_dict.items(),key=itemgetter(1),reverse=True)
    score_list_cut = score_list_sort[:10]
    sort_list = []
    for i in score_list_cut:
        sort_list += [i[0]]
    sort_list.sort()
    sentence_str = ''
    for i in sort_list:
        print(sentences_list[i])
        sentence_str += sentences_list[i]
    return sentence_str

@app.route("/")
def index():
    return render_template('zy.html')

@app.route("/zy",methods=["POST","GET"])
def zy():
    if request.method == 'GET':
        return render_template('zy.html')

    if request.method == 'POST':
        article = request.form.get('article')
        sentences_list, sentencelist = sentenceslist(article)
        sentence_vecs = sentence_to_vec(sentencelist)
        sentence_sort = sentences_sort(sentences_list, sentence_vecs)
        return render_template('zy.html',z_y = sentence_sort)

if __name__ == '__main__':

    app.run(debug=True)
