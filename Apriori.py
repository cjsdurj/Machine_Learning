def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def creatC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]);
    return map(frozenset,C1);
    

def scanD(D,ck,minsupport):
    ssCnt ={}
    for can in ck:     
        for transaction in D:
            if can.issubset(transaction):
                if not can in ssCnt.keys():ssCnt[can]=1;
                else:ssCnt[can]+=1
    numItems = float(len(D));
    retlist =[]
    supportData ={}
    for key in ssCnt.keys():
        support =ssCnt[key]/numItems
        if support >=minsupport:
            retlist.insert(0,key)
        supportData[key] =support
    return retlist,supportData; 


def aprioriGen(lk,k): #creates ck
    retList=[];
    lenlk=len(lk);
    for i in range(lenlk):
        for j in range(i+1,lenlk):
            l1=list(lk[i])[:k-2];
            l1.sort();
            l2=list(lk[j])[:k-2];
            l2.sort();
            if l1==l2:
                retList.append(lk[i] | lk[j])

    return retList;

def apriori(dataset,minsupport =0.5):
    c1=creatC1(dataset);
    D= list(map(set,dataset));
    L1,supportdata =scanD(D,c1,minsupport);
    L =[L1];
    k=2;
    
    while (len(L[k-2])>0):
        CK = aprioriGen(L[k-2],k)
        Lk,supK =scanD(D,CK,minsupport);
        supportdata.update(supK);
        L.append(Lk);
        k+=1;
    return L,supportdata
            
            


if __name__ =='__main__':
    
    L,supportdata=apriori(loadDataSet())
    print(L)
    print(supportdata)
  
    