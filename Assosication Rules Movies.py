#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:47:49 2022

@author: vasanthdhanagopal
"""

import pandas as pd
import matplotlib.pyplot as plt

movies=pd.read_csv('copy file path')
movies=movies.iloc[:,5:]

#Apriori Algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets=apriori(movies,min_support=0.05,use_colnames=True,max_len=3)
frequent_itemsets.sort_values('support',ascending=False,inplace=True)

plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11])
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)


########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
b = rules_no_redudancy.sort_values('lift',ascending=False).head(10)


import networkx as nx
xfig, ax=plt.subplots(figsize=(10,4))
GA=nx.from_pandas_edgelist(rules.iloc[:10,:],source='antecedents',target='consequents')
nx.draw(GA,with_labels=True)
plt.show()





