import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import jieba
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

def cut_word(word):  # 将字符串切成分词jieba
    cw = jieba.cut(word)
    return (' '.join(cw))
def load_stopwords(file_path):  # 将停用词转化成列表
    with open(file_path, encoding='UTF-8') as f:
        lines = f.readlines()
    words = []
    for line in lines:
        line = line.encode('unicode-escape').decode('unicode-escape')
        line = line.strip('\n')  # 除特定字符回车
        words.append(line)
    return words
# 特征提取方法 TF-IDF
def tfidf(data, stop_words):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.8,
                                       ngram_range=(1, 2))  # 计算TF-IDF，严格忽略高于给出阈值的文档频率的词条，提取的n-gram的n-values的下限和上限范围
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer
def data_process(n):
    data = pd.read_csv('./{}.csv'.format(n), encoding='utf-8', header=None, low_memory=False).astype(str)  # 读入第一步处理后的1~5列表
    data.columns = ['rating', 'comment']  # 确定列索引
    data["comment1"] = data['comment'].apply(cut_word)  # 列每一行进行cut_word函数操作

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data["comment1"],
                                                        data["rating"],
                                                        train_size=15000,
                                                        test_size=1000,
                                                        random_state=1)
    return X_train, X_test, y_train, y_test
def combine(X_train1, X_train2, X_train3):  # 把多个元素连接放到一起
    c = X_train1.append(X_train2)
    c1 = c.append(X_train3)

    return c1

'''
# 训练并保存模型
def SVMClassify(model, X_train_tfidf, y_train):
    if model == 'SVM':  # SVM 算法
        clf_tfidf = SVC(kernel="linear", C=16, gamma='auto', probability=True)
    elif model == 'LinearSVM':  # 线性 SVM 算法
        clf_tfidf = LinearSVC()
    elif model == 'nb':
        clf_tfidf = MultinomialNB()  # 朴素贝叶斯
    elif model == 'lr':
        clf_tfidf = LogisticRegression()  # 逻辑回归
    elif model == 'GBDT':
        clf_tfidf = GradientBoostingClassifier()  # 决策树
    elif model == 'RF':
        clf_tfidf = RandomForestClassifier(n_estimators=100)  # 随机森林

    elif model == 'MLP':# 神经网络
        clf_tfidf = MLPClassifier(hidden_layer_sizes=(400, 100), alpha=0.01, max_iter=300)

    clf_tfidf.fit(X_train_tfidf, y_train)  # 训练
    #joblib.dump(clf_tfidf, modelFile)  # 保存模型
    predict_test = clf_tfidf.predict(X_test_tfidf)
    accuracy = accuracy_score(predict_test, y_test)  # 分类准确率
    confusion = metrics.confusion_matrix(y_test, predict_test)  # 混淆矩阵
    result = classification_report(y_test, predict_test)  # 预测结果
    print("模型{}预测准确性：{}".format(model, accuracy))
    print("混淆矩阵:\n{}".format(confusion))
    print("预测结果:\n{}".format(result))
'''

if __name__ == '__main__':
    # 通过pandas读入数据
    X_train1, X_test1, y_train1, y_test1 = data_process(1)
    X_train2, X_test2, y_train2, y_test2 = data_process(0)
    X_train3, X_test3, y_train3, y_test3 = data_process(-1)
    X_train = combine(X_train1, X_train2, X_train3)
    X_test = combine(X_test1, X_test2, X_test3)
    y_train = combine(y_train1, y_train2, y_train3)
    y_train = y_train.astype(float)  # 评分，浮点型
    y_test = combine(y_test1, y_test2, y_test3)
    y_test = y_test.astype(float)
    modelFile = "Model"
    stop_words = load_stopwords('stop_word.txt')
    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train, stop_words)
    # print(X_train_tfidf)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)  # 归一化
    # 预测测试集结果
    for i in range(2):
        if i == 0:
            clf_tfidf = MultinomialNB()  # 朴素贝叶斯
            clf_tfidf.fit(X_train_tfidf, y_train)  # 训练
            #joblib.dump(clf_tfidf, modelFile)  # 保存模型
            predict_test = clf_tfidf.predict(X_test_tfidf)
            accuracy = accuracy_score(predict_test, y_test)  # 分类准确率
            confusion = metrics.confusion_matrix(y_test, predict_test)  # 混淆矩阵
            result = classification_report(y_test, predict_test)  # 预测结果
            print("模型{}预测准确性：{}".format("NATIVE BYES", accuracy))
            print("混淆矩阵:\n{}".format(confusion))
            print("预测结果:\n{}".format(result))
        if i == 1:
            clf_tfidf = MLPClassifier(hidden_layer_sizes=(400, 100), alpha=0.01, max_iter=300)
            clf_tfidf.fit(X_train_tfidf, y_train)  # 训练
            #joblib.dump(clf_tfidf, modelFile)  # 保存模型
            predict_test = clf_tfidf.predict(X_test_tfidf)
            accuracy = accuracy_score(predict_test, y_test)  # 分类准确率
            confusion = metrics.confusion_matrix(y_test, predict_test)  # 混淆矩阵
            result = classification_report(y_test, predict_test)  # 预测结果
            print("模型{}预测准确性：{}".format("MLP", accuracy))
            print("混淆矩阵:\n{}".format(confusion))  
            print("预测结果:\n{}".format(result))

