import matplotlib.pylab as plt
import numpy as np

def loadDataSet():
    # 创建数据列表
    dataMat=[]
    # 创建标签列表
    labelMat=[]
    
    fr=open('testSet.txt')
   
    for line in fr.readlines():
       
        lineArr=line.strip().split()
        
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

"""
函数说明：绘制数据集
"""
def plotDataSet():
    
    dataMat,labelMat=loadDataSet()

    #转化成numpy的array
    dataArr=np.array(dataMat)
    
    n=np.shape(dataMat)[0]
    #正样本
    xcord1=[];ycord1=[]
    #负样本
    xcord2=[];ycord2=[]
    #根据数据集标签进行分类
    for i in range(n):
        #1为正样本
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        #0为负样本
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    
    fig=plt.figure()
    #添加subplot
    ax=fig.add_subplot(111)
    #绘制正样本
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)
    #绘制负样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)

    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    #转化成numpy的mat
    dataMatrix=np.mat(dataMatIn)
    
    # 转化成numpy的mat，并进行转置
    labelMat=np.mat(classLabels).transpose()
    #返回dataMatrix的大小，m为行，n为列
    #print(np.shape(dataMatrix))
   
    m,n=np.shape(dataMatrix)
    #移动步长，也就是学习速率，控制参数更新的速度
    alpha=0.001
    #最大迭代次数
    maxCycles=500
    weights=np.ones((n,1))
    
    #print(weights)
    
    for k in range(maxCycles):
        #梯度上升矢量化公式
        h=sigmoid(dataMatrix*weights)
        
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    #将矩阵转化为数组，并返回权重参数
    return weights.getA()

def plotBestFit(weights):
    #加载数据集
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    #数据个数
    n=np.shape(dataMat)[0]
    #正样本
    xcord1=[]
    ycord1=[]
    #负样本
    xcord2=[]
    ycord2=[]
    #根据数据集标签进行分类
    for i in range(n):
        #1为正样本
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        #0为负样本
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    x=np.arange(-3.0,3.0,0.1)

    y=(-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)

    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()




if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    weights=gradAscent(dataMat,labelMat)
    plotBestFit(weights)
