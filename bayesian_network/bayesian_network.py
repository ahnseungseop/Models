# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def get_order(structure):
    # structure: dictionary of structure
    # key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    key = list(structure.keys())
       
    var_order=[]
    for i in range(0, len(key)):
        if len(structure[key[i]])==0 :
            var_order.append(key[i])
        
    key=[x for x in key if x not in var_order]
    
    for i in range(0, len(key)):
        if structure[key[i]]==var_order :
            var_order.append(key[i])
            
    
    key=[x for x in key if x not in var_order]
    
    for i in range(0, len(key)):
        if set(structure[key[i]])==set(var_order) :
            var_order.append(key[i])
            
    key=[x for x in key if x not in var_order]
    
    for i in range(0, len(key)):
        if set(structure[key[i]]).issubset(set(var_order)) :
            var_order.append(key[i])
    
    key=[x for x in key if x not in var_order]
        
    for i in range(0, len(key)):
        if set(structure[key[i]]).issubset(set(var_order)) :
            var_order.append(key[i])
    
    
    
    return var_order
    

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
 
        
    parms={}
    
    parms={}
    for i in range(0,len(structure)):
        if len(structure[var_order[i]])==0 :
                parms[var_order[i]]=data.groupby(var_order[i]).count()['index']/len(data) 
        else :
             b=structure[var_order[i]]
             parms[var_order[i]]=(data.groupby([element for array in [var_order[i],structure[var_order[i]]] for element in array]).count() 
             / data.groupby(b).count())['index']  
    return parms    
            
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        if len(parms[var])<= 3 :
            print(parms[var].to_frame().reset_index().T)
        else :
            col=parms[var].to_frame().reset_index().columns.tolist()
            index=list(set(col)-set([var,'index']))
            a=parms[var].to_frame().reset_index().set_index(index[0])
                        
            print(a.pivot_table(a, index=index,columns=var))
        
        
        
        
        
    
data=pd.read_csv('bayesian_data', sep=' ') #파일경로수정
data['index']=range(0,500)
str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')


str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')