"""Adaptive Test Recommendation Experiments (IRT-based simulator)

This script answers RQ2.b by simulating learners using a neural IRT model
from EduCDM (GDIRT). It evaluates the same material selection strategies as
the BKT-based setup and records results. Requires PyTorch and EduCDM.
"""

import pandas as pd
import numpy as np
from pyBKT.models import Model
import scipy.sparse as sparse

from torch.utils.data import TensorDataset, DataLoader
import torch

from EduCDM import GDIRT
from utils import *
from random import *
import time


# Data
data = pd.read_csv('./Data/mat_all.csv')
data_2 = pd.read_csv('./Data/matmat/answers.csv')

tests2diff = data[data.type=='test'][['id','difficulty']].set_index('id').T.to_dict('list')
diff2tests = data[data.type=='test'][['id','difficulty']].groupby('difficulty')['id'].apply(list).reset_index().set_index('difficulty').T.to_dict()
diff2tests = {key:value['id'] for key,value in diff2tests.items()}

difficulties = sorted(list(diff2tests.keys()))
diff_hier = dict()

len_ = len(difficulties)

for i,j in zip(range(len_-1),range(1,len_)):
    ele_1 = difficulties[i]
    ele_2 = difficulties[j]
    
    if i == 0:
        diff_hier[ele_1] = [ele_2]
    else:
        prec_ele = difficulties[i-1]
        diff_hier[ele_1] = [prec_ele,ele_2]
        
        if i == len_-2:
            diff_hier[ele_2] = [ele_2]

data = data[data.type=='test']
data = data[['id','difficulty']].rename(columns={'id':'item'})

data_2 = data_2[['id','item','student','correct']]
data_2 = pd.merge(data_2, data, on='item')

df_difficulties_first = data[['difficulty']].drop_duplicates()

diff_2_id = df_difficulties_first.copy()
diff_2_id['id'] = range(len(diff_2_id))
diff_2_id = diff_2_id.set_index('difficulty').T.to_dict('list')

methods2function = {
    'RANDOM_Alternate': get_alternate_random_material,
    'MOAE': get_mo_apt_exp_materials,
    'MOEG': get_mo_exp_gap_materials,
    'MOAG': get_mo_apt_gap_materials,
    'MOO': get_mo_3_materials,
}

letter2values = {
    'e':[0, 0.44],
    'm':[0.44, 0.7],
    'h':[0.7, 1]
}

def transform(x, y, z, batch_size, **params):
    """Create a DataLoader from indices x, item ids y, and labels z."""
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

batch_size = 32

for k in [3]:
    df_results = pd.DataFrame(columns=['student','method','skill_student','Materials','Answers','new_skill_student','step','time','learns_proba'])

    cpt_users = 0

    liste = list(data_2.student.unique())[:100]

    for user in liste:
        df_train_first = pd.DataFrame(columns=data_2.columns)
        data_inter = data_2[data_2.student == user]
        
        data_inter = data_2[(data_2.student == user) & (data_2.difficulty < 0.2)]
        
        if len(data_inter) == 0:
            df_train_first = data_2[(data_2.difficulty < 0.2) & (data_2.correct==0)].sample(frac=1).head(2)
            df_train_first = pd.concat([df_train_first,data_2[(data_2.difficulty < 0.2) & (data_2.correct==1)].sample(frac=1).head(2)])
            df_train_first.student = user
            skill_user = 0.13
        else:
            df_train_first = data_inter.sample(frac=1).head(4)
            if len(df_train_first[df_train_first.correct==1]) > 0 :
                skill_user = 0.13
            else:
                skill_user = 0.12
        
        df_train_first = df_train_first.sort_values('id')
        seen = df_train_first[['item','difficulty','correct']].values.tolist()
        
        df_difficulties_first = df_difficulties_first[df_difficulties_first.difficulty > skill_user]
        diff2apt, diff2exp, diff2gap = get_sets(df_difficulties_first, skill_user, seen, windows_l=k)
        
        df_difficulties_first['apt'] = df_difficulties_first.difficulty.apply(lambda x: diff2apt[x])
        df_difficulties_first['exp'] = df_difficulties_first.difficulty.apply(lambda x: diff2exp[x])
        df_difficulties_first['gap'] = df_difficulties_first.difficulty.apply(lambda x: diff2gap[x])

        skill_user_first = skill_user

        stu_num, prob_num = 1, 42
        
        for method, func in methods2function.items():
            df_train = df_train_first.copy()
            df_difficulties = df_difficulties_first.copy()
            skill_user = skill_user_first
            ncc = {}
            
            step = 0
            score = 0

            alternate = ['e','e','m','m','h','h']

            df_train['diff_2'] = df_train.difficulty.apply(lambda x: diff_2_id[x][0])
            df_train = df_train[['item','difficulty','correct','diff_2']]
            
            while skill_user < 0.99 and step < 500:
                print(f'User : {cpt_users}\tMethod : {method}\tskill user : {skill_user}')

                result_row = [user,method,skill_user]
                
                df_train.index = range(len(df_train))
                train = transform(len(df_train)*[1], df_train["diff_2"], df_train["correct"], batch_size)

                cdm = GDIRT(2,prob_num)
                cdm.train(train, epoch=100)
                
                beg_time = time.time()

                #Choice of materials
                if method == 'RANDOM_Alternate':
                    df_test = func(data, skill_user, alternate, letter2values, step, k)
                else:
                    df_test = func(data, df_difficulties, skill_user, diff2tests, diff_hier, k=k)
                
                end_time = time.time() - beg_time
                
                df_test = df_test[['item','difficulty']]

                if len(df_test) == 0:
                    m = 0
                elif len(df_test) < k:
                    print(f'{skill_user}  -  {len(df_test)}')
                    inter = df_test.head(1)
                    while len(df_test) < k:
                        df_test = df_test.append(inter)
                
                m = min(k,len(df_test))

                df_test['diff_2'] = df_test.difficulty.apply(lambda x: diff_2_id[x][0])
                df_test['correct'] = [1 for i in range(m)]
                df_test.index = range(len(df_test))
                test = transform(len(df_test)*[1], df_test["diff_2"], df_test["correct"], batch_size)

                #GET ADUP MATERIALS       
                df_pred = cdm.eval_prediction(test)
                df_test['correct'] = df_pred
                df_test['correct'] = df_test.correct.apply(lambda x: 1 if x > 0.9 else 0)

                answers = list(df_test['correct'])

                #Skill update
                #new_skill_user, ncc = update_skill(skill_user, df_test[['difficulty','correct']], ncc, 3)
                new_skill_user = update_skill_2(skill_user, df_test[['difficulty','correct']], ncc)
                
                result_row.append(list(df_test.difficulty))

                df_test = df_test[['item','difficulty','correct','diff_2']]
                df_test = pd.concat([df_train, df_test], ignore_index=True, sort=False)
                df_train = df_test.copy()

                if 'RANDOM' not in method:
                    seen = df_train[['item','difficulty','correct']].values.tolist()

                    if new_skill_user != skill_user:
                        df_difficulties = df_difficulties[df_difficulties.difficulty > new_skill_user]
                        diff2apt, diff2exp, diff2gap = get_sets(df_difficulties, skill_user, seen, windows_l=k)

                        df_difficulties['apt'] = df_difficulties.difficulty.apply(lambda x: diff2apt[x])
                        df_difficulties['exp'] = df_difficulties.difficulty.apply(lambda x: diff2exp[x])
                        df_difficulties['gap'] = df_difficulties.difficulty.apply(lambda x: diff2gap[x])

                    else:
                        diff2apt, diff2exp, diff2gap = get_sets(df_difficulties, skill_user, seen, windows_l=k)

                        df_difficulties['exp'] = df_difficulties.difficulty.apply(lambda x: diff2exp[x])
                        df_difficulties['gap'] = df_difficulties.difficulty.apply(lambda x: diff2gap[x])
                
                elif method == 'RANDOM_Alternate':
                    if new_skill_user > 0.42:
                        alternate = ['m', 'm', 'h', 'h']
                    elif new_skill_user >= 0.7:
                        alternate = ['h', 'h']
                
                skill_user = new_skill_user

                params = cdm.irt_net

                result_row.append(answers)
                result_row.append(skill_user)
                result_row.append(step)
                result_row.append(end_time)
                result_row.append(params)

                step += 1

                df_results.loc[len(df_results)] = result_row

        cpt_users += 1

        df_results.to_csv(f'irt_results_{k}_ncc_1.csv', index=False)
    
    df_results.to_csv(f'irt_results_{k}_ncc_1.csv', index=False)
