#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pickle
from model.IFP import reference_atom, get_Molecules, AAIFP_class


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[ ]:


df_Interaction = load_obj('df_Interaction')


# In[6]:


# reference_list = reference_atom(df_Interaction)
reference_atom = load_obj('refer_atoms_list')
reference_res = load_obj('refer_res_list')
molecule_list = get_Molecules(df_Interaction)
print(f"\nNumber of the reference atoms are :{len(reference_atom)}\n")
print(f"The reference atoms are :\n{reference_atom}\n")
print(f"\nNumber of the reference reses are :{len(reference_res)}\n")
print(f"The reference reses are :\n{reference_res}\n")


# In[7]:


AAIFP = AAIFP_class(df_Interaction, reference_atom, reference_res)
AA_full, RES_full = AAIFP.calIFP()

# print(AA_full)
# print(RES_full)


# In[ ]:


AA_full.to_csv('AAIFP_full.csv')
RES_full.to_csv('RESIFP_full.csv')


# In[ ]:
