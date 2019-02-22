
# coding=utf-8

import numpy as np

from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn import svm

from sklearn.preprocessing import StandardScaler
class BaggingSVM(object):

    def __init__(self, bootstrap_samples_num, bootstrap_loop_num, C,ture=[],cal_weight=[]):




        #样本数目
        self.bootstrap_samples_num = bootstrap_samples_num
        # 抽样次数（基分类器个数）
        self.bootstrap_loop_num = bootstrap_loop_num
        # 惩罚因子
        self.C = C
        # self.scaler = StandardScaler()
        self.ture=ture
        # cx 记录基分类器权重
        self.cal_weight = cal_weight
        # cx 记录已有样本
        self.samples=np.array([])





    def Bootstrap(self, X, Y, weight):

        """

                Do Bootsrap sampling for data X and Y.
                从XY进行抽样



                Parameters

                ----------

                 X:  array of shape [n_samples, n_features]

                    The samples of data.



                Y:  array of shape [n_samples, ]

                    The labels of data.





                Returns

                -------

                X_in_bag: dict

                    The samples of different bootstrap loop in bag.



                Y_in_bag: dict

                    The lables of different bootstrap loop in bag.



                X_out_bag: dict

                    The samples of different bootstrap loop out bag.



                Y_out_of_bag: dict

                    The lables of different bootstrap loop in bag.



                """



        # 样本个数
        m = X.shape[0]

        # 每个样本的特征个数（维度）
        n = X.shape[1]



        X_in_bag = defaultdict()

        Y_in_bag = defaultdict()

        X_out_bag = defaultdict()

        Y_out_bag = defaultdict()

        W_in_bag = defaultdict()

        W_out_bag = defaultdict()




        ll_indx = np.random.randint(0, m, size=(self.bootstrap_loop_num, m))

        for loop in range(self.bootstrap_loop_num):

            X_in_bag[str(loop)] = X[ll_indx[loop]]

            Y_in_bag[str(loop)] = Y[ll_indx[loop]]

            W_in_bag[str(loop)] = weight[ll_indx[loop]]

            X_out_bag[str(loop)] = np.delete(X, list(set(ll_indx[loop])), 0)

            Y_out_bag[str(loop)] = np.delete(Y, list(set(ll_indx[loop])))

            W_out_bag[str(loop)] = np.delete(weight, list(set(ll_indx[loop])))



        return X_in_bag, Y_in_bag, X_out_bag, Y_out_bag, W_in_bag ,W_out_bag



    def fit(self, X, Y, weight):

        """

        Train the WeightedCeBagSVM for dataset.



        Parameters

        ----------

        X: array of shape [n_samples, n_features]

            The samples of dataset.



        Y: array of shape [n_samples, ]

            The lables of dataset.

        """
        # print(weight)
        self.samples=np.append(self.samples, X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)


        # print('fit:',X.shape)
        # cx 增加样本权重
        self.X_in_bag, self.Y_in_bag, self.X_out_bag, self.Y_out_bag , W_in_bag ,W_out_bag= self.Bootstrap(X, Y,weight)

        self.clf = defaultdict()
        sv=[]
        for loop in range(self.bootstrap_loop_num):

            # self.clf[str(loop)] = SVC(C=self.C).fit(self.X_in_bag[str(loop)],self.Y_in_bag[str(loop)])
            # cx 训练分类器时加入样本权重
            self.clf[str(loop)] = SVC(C=self.C).fit(self.X_in_bag[str(loop)], self.Y_in_bag[str(loop)], W_in_bag[str(loop)])
            # 训练得到分类器
            #print('SV',self.clf[str(loop)].support_,len(self.clf[str(loop)].support_))  # 获取支持向量（某一基分类器）
            sv.append(list(self.clf[str(loop)].support_))
            # 将所有基分类器的支持向量整合在一起
        sv_idx = []
        for id in sv:
            if id not in sv_idx:
                sv_idx.append(id)
        #print('sv_idx', sv_idx)
        # cx 去掉支持向量集中重复部分

        for loop in range(self.bootstrap_loop_num):
            pre = []
            # 预测结果
            t=0

            pre.append(self.clf[str(loop)].predict(self.X_out_bag[str(loop)]))
           # print('pre',len(pre[0]))
           # print(len(self.Y_out_bag[str(loop)]))
            for i in range(len(self.Y_out_bag[str(loop)])):
                print(self.Y_out_bag[str(loop)][i])
                # print('a',pre[0][i])

                if (self.Y_out_bag[str(loop)][i]) == pre[0][i]:
                    t=t+1
                # Y=np.array(self.Y_out_bag[str(loop)])
                # print(Y[i])
             #   print('666',pre[0][i])
            print(t / len(self.Y_out_bag[str(loop)]))
            self.ture.append(t / len(self.Y_out_bag[str(loop)]))
            # 逐个记录基分类器的分类准确率 self.ture
            # cx 计算各分类器权重（初始为训练时的准确率）
        self.cal_weight=self.ture
        print('yuce')

    # def input_newfeaure(self,newX, newY, newweight):
    #     # 特征筛选+基分类器权重调整+训练新的基分类器
    #     Newdata_Predictions=self.predict(newX,c=0)
    #     falseIndex = []
    #     falsefeaure = []
    #     print('Newdata_Predictions.shape[0]',Newdata_Predictions.shape[0])
    #     for i in range(newY.shape[0]):
    #         # 筛选不能被正确分类的新增特征
    #         print('newY[0][i]',newY[i])
    #         print('Newdata_Predictions[0][i]',Newdata_Predictions[i])
    #         if newY[i]!=Newdata_Predictions[i]:
    #             falseIndex.append(i)
    #             # 不能被正确分类的增量特征的索引
    #             falsefeaure.append(newX[1])
    #             # 不能被正确分类的增量特诊
    #     # print('falsefeaure',falsefeaure)
    #     kldiv=[]
    #     for i in range(newX.shape[0]):
    #         # 遍历新增特征
    #         min=0
    #         # 差异越小kl散度值越小，所以选择最小值
    #         for j in range(self.samples.shape[0]):
    #             k=kl.symmetricalKL(newX[i], self.samples[j])
    #             if k<min:
    #                 min=k
    #         kldiv.append(min)
    #     # print('kldiv',kldiv)
    #     for i in range(len(newX.shape[0])):
    #         if kldiv[i]>-11:
    #             newweight[i]=0












    def predict(self, X,c=0.66):

        """

        Predict the lables of given samples.



        Parameters

        ----------

        X: array of shape [n_samples, n_features]



        Returns

        -------

        Predictions: array of [n_samples, ]

            The predicted lables of given samples.

        """
        self.scaler = StandardScaler()
        self.scaler.fit(X)


        X = self.scaler.transform(X)

        P = []

        # print("Predict:", X.shape)
        s=0
        classA=0
        classB=0
        for loop in range(self.bootstrap_loop_num):
            if self.ture[loop]>=c:
                # 选择训练正确率大于0.66的基分类器
                P.append(self.clf[str(loop)].predict(X))
                print('self.clf[str(loop)].predict(X)',self.clf[str(loop)].predict(X))
                # p记录个分类器识别结果
                s=s+1


        a=0
        # np.matrix(P).T是各基分类器识别结果矩阵的转置
        for i in np.matrix(P).T:
             print('aaaa',i)
             a=a+1
        print('ppp',a)
        print(s)
        print(self.bootstrap_loop_num)
        Predictions = [0 if np.sum(i) < s / 2.0 else 1 for i in  np.matrix(P).T]




        return np.array(Predictions)






    def score(self,X, Y):

        """

        Compute the accuracy of classifier on given samples.



        Parameters

        ----------



        X: array of shape [n_samples, n_features]

            Given samples.

        Y: array of shape [n_samples, n_features]

            Given lables.



        Returns

        -------

        acc: float

            The accuracy of classifier on given samples.

        """



        Y_pred = self.predict(X)

        return np.sum(np.array(Y) == np.array(Y_pred)) / len(Y)