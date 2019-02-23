
import numpy as np;

from functools import reduce


def loadDataSet():
    # 切分的词条
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec




def CreateDictionory(documents):
    Dictionary =set([]);
    
    for doc in documents:
        Dictionary = Dictionary|set(doc);
    
    dicList = list( Dictionary);
    dicList.sort();
    return dicList;



def  Doc2Vec(dictionary,doc):
    Vec =[0]*len(dictionary);
    
    for word in doc:
        Vec[dictionary.index(word)] +=1;
            
    return Vec;

        
def trainNB0(trainMtrix,trainCategory):
    
    numTrainDocs=len(trainMtrix)
    
    numWords=len(trainMtrix[0])
    
    pAbusive=sum(trainCategory)/float(numTrainDocs)
   
    p0Num=np.ones(numWords);p1Num=np.ones(numWords)
   
    p0Denom=2.0;p1Denom=2.0

    for i in range(numTrainDocs):
      
        if trainCategory[i]==1:
            p1Num+=trainMtrix[i]
            p1Denom+=sum(trainMtrix[i])
       
        else:
            p0Num+=trainMtrix[i]
            p0Denom+=sum(trainMtrix[i])
    
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom
    
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    p1=reduce(lambda x,y:x*y,vec2Classify*p1Vec)*pClass1
    p0=reduce(lambda x,y:x*y,vec2Classify*p0Vec)*(1.0-pClass1)'''
    
    p1 =sum(vec2Classify*p1Vec)+np.log(pClass1);
    p0 =sum(vec2Classify*p0Vec)+ np.log(1.0-pClass1);
    
    print('p0:',p0)
    print('p1:',p1)
    if p1>p0:
        return 1
    else:
        return 0
    
    

        





if __name__=='__main__':
    postingList,classVec=loadDataSet();
    dict0=CreateDictionory(postingList);
    
    print(dict0.index('stupid'));
    trainset =[Doc2Vec(dict0,doc) for doc in postingList];
    
    p0Vect,p1Vect,pAbusive=trainNB0(trainset,classVec);
    
     #测试样本1
    testEntry=['love','my','dalmation']
   
    thisDoc=np.array(Doc2Vec(dict0,testEntry))
   
    if classifyNB(thisDoc,p0Vect,p1Vect,pAbusive):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
    
    #测试样本2
    testEntry=['stupid','garbage']
    thisDoc = np.array(Doc2Vec(dict0, testEntry))
    if classifyNB(thisDoc,p0Vect,p1Vect,pAbusive):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')  

