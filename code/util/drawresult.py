import numpy as np
from wsfx2.code.util.excel_op import createx
f1 = open('/Users/wenny/nju/task/ppt/record_alpha.txt','r',encoding='utf-8')
lines = f1.read().split('\n')
array = []
for line in lines:
    line = line.strip()
    line = list(filter(lambda x:x.strip()!='',line.split(' ')))
    line = list(map(float,line))
    array.append(line)

array = np.array(array[:-1])
meanrow = np.mean(array,axis=1)
meanrow = np.expand_dims(meanrow,axis=1)
print(meanrow)
rows = [i for i in range(30)]
createx('drawlawweight_alpha',rows= rows, colums=['weight'],dir='../',data=meanrow)


