# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:23:47 2018

@author: Dell
"""

'''
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
'''
from math import log
import operator

def creatDataSet():
    
    dataset=[[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels=['年龄','有工作','有自己的房子','信贷情况']
    
    return dataset,labels


def calc_empirical_entropy(dataset):
    numEntries =len(dataset)
    labelCounts={}
    
    for data in dataset:
        currentlabel =data[-1]
        
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel]=0;
        labelCounts[currentlabel]+=1;
    
    empirical_entropy =0.0
    
    for key in labelCounts.keys():
        prob =float(labelCounts[key])/numEntries
        empirical_entropy -=prob*log(prob,2)
    return empirical_entropy


def splitDataset(dataset,axis,value):
    retDataSet=[]
    for item in dataset:
        if item[axis]==value:
            reduceditem =item[:axis]
            reduceditem.extend(item[axis+1:])
            retDataSet.append(reduceditem)
    return retDataSet


def choose_BestFeature_To_Split(dataset):
    numFeature =len(dataset[0])-1
    baseEntropy = calc_empirical_entropy(dataset)
    
    bestInfoGain=0.0
    bestFeature=-1
    
    for i in range(numFeature):
        featlist=[example[i] for example in dataset]
        uniquevalues=set(featlist)
        
        newEntropy =0.0
        for value in uniquevalues:
            subdataset =splitDataset(dataset,i,value)
            newEntropy+=float(len(subdataset))/len(dataset)*calc_empirical_entropy(subdataset)
        
        infoGain = baseEntropy-newEntropy
        print('第%d个特征的信息增益为%.3f'%(i,infoGain))
        
        if infoGain>bestInfoGain:
            bestFeature=i
            bestInfoGain=infoGain
    return bestFeature


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys:
            classCount[vote]=0
        classCount[vote]+=1
    
    sortedClassCount =sorted(classCount.items(),key=operator.itemgetter(1),reverse =True)
    return sortedClassCount[0][0]

def createTree(dataset,labels,featlabels):
    classList =[example[-1] for example in dataset]
    
    if classList.count(classList[0])==len(classList):
        return classList[0]
    
    if len(dataset[0])==1:
        return majorityCnt(classList)
    
    bestFeat =choose_BestFeature_To_Split(dataset)
    
    bestFeatlabel =labels[bestFeat]
    featlabels.append(bestFeatlabel)
    
    mytree ={bestFeatlabel:{}}
    del(labels[bestFeat])
    uniqueVals =set([example[bestFeat] for example in dataset])
    
    for value in uniqueVals:
        mytree[bestFeatlabel][value]=createTree(splitDataset(dataset,bestFeat,value),
                                               labels,featlabels)
    return mytree

 
    
if __name__ =="__main__":
    dataset,labels =creatDataSet();
    #entropy = calc_empirical_entropy(dataset)
    #print('最优的索引值为：'+str(choose_BestFeature_To_Split(dataset)))
    
    #print(dataset)
    #print(entropy)
    print(createTree(dataset,labels,[]))
    
    



