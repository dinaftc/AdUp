import pandas as pd
import numpy as np
import scipy.sparse as sparse
import time

from random import *
from pyBKT.models import Model
from utils import *
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


defaults = {'order_id': 'id',
            'skill_name': 'difficulty',
            'user_id': 'student',
            'correct': 'correct', 
            'multigs':'difficulty'}

df_difficulties_first = data[['difficulty']].drop_duplicates()

methods2function = {
    'RANDOM_Alternate': get_alternate_random_material,
    'MOAE': get_mo_apt_exp_materials,
    'MOEG': get_mo_exp_gap_materials,
    'MOAG': get_mo_apt_gap_materials,
    'MOO': get_mo_3_materials,
    'MOA': get_apt_materials,
    'MOE': get_exp_materials,
    'MOG': get_gap_materials,
}

letter2values = {
    'e':[0, 0.44],
    'm':[0.44, 0.7],
    'h':[0.7, 1]
}

ncc_repeat = 1

for k in [3,5]:
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
        
        for method, func in methods2function.items():
            df_train = df_train_first.copy()
            df_difficulties = df_difficulties_first.copy()
            skill_user = skill_user_first
            ncc = {}
            
            max_id = data_2.id.max()+1
            step = 0

            alternate = ['e','e','m','m','h','h']
            
            while skill_user < 0.99 and step < 500:
                print(f'User : {cpt_users}\tMethod : {method}\tskill user : {skill_user}')

                row_ids = list(df_train.id)
                result_row = [user,method,skill_user]

                model = Model(seed=user)
                model.fit(data = df_train, defaults = defaults)
                
                beg_time = time.time()

                df_train.difficulty = df_train.difficulty.astype(float)
                #Choice of materials
                if method == 'RANDOM_Alternate':
                    df_test = func(data, skill_user, alternate, letter2values, step, k)
                elif method in ['MOA','MOE','MOG']:
                    df_test = func(data, df_difficulties, skill_user, k)
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

                #GET MATERIALS       
                df_test['student'] = [user for i in range(m)]
                df_test['correct'] = [1 for i in range(m)]
                df_test['id'] = [max_id+i for i in range(m)]
                df_test = df_test[['id','item','student','correct','difficulty']]

                max_id += m
                result_row.append(list(df_test.difficulty))

                df_test = pd.concat([df_train, df_test], ignore_index=True, sort=False)

                df_pred = model.predict(data = df_test.copy())
                df_pred['correct'] = df_pred.state_predictions.apply(lambda x: 1 if x >= 0.7 else 0)

                answers = list(df_pred[~df_pred.id.isin(row_ids)].correct)
                df_test.loc[~df_test.id.isin(row_ids),'correct'] = answers

                #Skill update
                new_skill_user, ncc = update_skill(skill_user, df_test[~df_test.id.isin(row_ids)], ncc, ncc_repeat) # NCC > 1
                #new_skill_user = update_skill_2(skill_user, df_test[~df_test.id.isin(row_ids)], ncc) # NCC == 1
                
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
                
                params = model.params().reset_index()
                params = params.drop(columns=['class']).set_index(['skill','param']).value.T.to_dict()

                result_row.append(answers)
                result_row.append(skill_user)
                result_row.append(step)
                result_row.append(end_time)
                result_row.append(params)

                step += 1

                df_results.loc[len(df_results)] = result_row

        cpt_users += 1

        df_results.to_csv(f'bkt_results_{k}_ncc_{ncc_repeat}.csv', index=False)
    
    df_results.to_csv(f'bkt_results_{k}_ncc_{ncc_repeat}.csv', index=False)
