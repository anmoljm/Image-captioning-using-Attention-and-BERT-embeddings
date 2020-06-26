aList = [123, 'xyz', 'zara','xyz', 'abc']; 
indices = [i for i, x in enumerate(aList) if x == "xyz"]
print(indices)
import os
adict ={'A':13, 'B':14, 'C':15}

inverted_dict = dict([[v,k] for k,v in adict.items()])
cap_idx = [13,15]
cap = ' '.join([inverted_dict[word_idx] for word_idx in cap_idx])
print(cap)
