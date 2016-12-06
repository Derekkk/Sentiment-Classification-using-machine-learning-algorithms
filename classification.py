# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.lda import LDA

'''导入txt'''
Data_label = 'amazon_yelp_imdb_label.txt'
twitterData = 'amazon_yelp_imdb.txt'

#Data_label = 'Multi_label.txt'
#twitterData = 'Multi_data.txt'

def tweet_dict(twitterData):
    twitter_list_dict = []
    Data=open(twitterData)
    for line in Data:
        line_1=line.lstrip(' ')
        twitter_list_dict.extend([line_1])
    return twitter_list_dict

def labels(Data_label):
    label_list = []
    Data_2=open(Data_label)
    for line2 in Data_2:
        #label_list.extend([int(line2)])
        label_list.extend([line2])
    return label_list

tweets = tweet_dict(twitterData)
label = labels(Data_label)



'''''加载数据集'''
movie_reviews = ['shit.', 'waste my money.', 'waste of money.', 'sb movie.', 'waste of time.', 'a shit movie.', 'nb! nb movie!', 'nb!', 'worth my money.', 'I love this movie!', 'a nb movie.', 'worth it!']
labels = ['neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos']

'''分割数据集'''
doc_terms_train, doc_terms_test, y_train, y_test = train_test_split(tweets, label, test_size=0.2)
print 'length of traing data:',len(doc_terms_train),'length of test data:',len(doc_terms_test)

'''''方法一'''
count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english', ngram_range=(1,1),smooth_idf=True)
x_train = count_vec.fit_transform(doc_terms_train)
x_test = count_vec.transform(doc_terms_test)
x = count_vec.transform(movie_reviews)
y = labels
print 'the length of the feature',len(count_vec.get_feature_names())
'''方法二'''

transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(count_vec.fit_transform(doc_terms_test))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=count_vec.get_feature_names()#获取词袋模型中的所有词语，等于count_vec.get_feature_names()
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

doc_test_vec=count_vec.fit_transform(doc_terms_test)
doc_train_vec=count_vec.fit()


'''计算相似度'''
'''
print(doc_terms_train)
print(count_vec.get_feature_names())
print(doc_terms_test)
print doc_terms_train

X=np.array(x_train.toarray())
Y=np.array(x_test.toarray())

print X.dot(Y.T)  #m*n矩阵，表示train中第n个数据和test中第m个数据的相似度

#计算相似度
SimMatrix = (tfidf * tfidf.T).A
print SimMatrix #"第一篇与第4篇的相似度"
'''
#print doc_terms_test
#print y_test

######################################################
#Multinomial Naive Bayes Classifier
print '*************************\nNaive Bayes\n*************************'
#create the Multinomial Naive Bayesian Classifier
clf = MultinomialNB(alpha = 0.01)
clf.fit(x_train,y_train);
pred_NB = clf.predict(x_test);
print 'accuracy:', clf.score(x_test,y_test)

######################################################
#KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
print '*************************\nKNN\n*************************'
knnclf = KNeighborsClassifier()#default with k=5
knnclf.fit(x_train,y_train)
pred_KNN = knnclf.predict(x_test);

print 'accuracy:', knnclf.score(x_test,y_test)


######################################################
#SVM Classifier
print '*************************\nSVM\n*************************'
svclf = SVC(kernel = 'linear')#default with 'rbf'
svclf.fit(x_train,y_train)
pred_SVM = svclf.predict(x_test);

print 'accuracy:', svclf.score(x_test,y_test)

