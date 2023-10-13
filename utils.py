import pandas as pd
import numpy as np
from random import *

def update_skill(skill_learner, df_materials, ncc, window = 3):
    max_skill = skill_learner
    enter = True
    
    for row in df_materials.itertuples():
        diff = row[5]
        correct = row[4]
        
        if diff in ncc:
            if len(ncc[diff])==window:
                ncc[diff].pop(0)
            
            ncc[diff].append(correct)
        else:
            ncc[diff] = [correct]
        
        if correct == 1 and enter:
            max_skill = diff
        elif enter:
            smallest_false_diff = diff
            enter = False

    if enter == False:
        for key,val in ncc.items():
            if key >= smallest_false_diff:
                ncc[key] = []
    
    if max_skill != skill_learner:
        if len(ncc[max_skill]) < window or sum(ncc[max_skill]) < window:
            max_skill = skill_learner
            
    return max_skill, ncc

def update_skill_2(skill_learner, df_materials, ncc, window = 3):
    max_skill = skill_learner
    
    for row in df_materials.itertuples():
        diff = row[5]
        correct = row[4]
        
        if correct == 1:
            max_skill = diff
        else:
            break
            
    return max_skill

def apt(dif_material, skill_learner):
    diff = dif_material - skill_learner
    
    if diff < 0:
        return 0
    
    return diff

def exPerf(dif_material, seen_materials, sim_threshold=0.1):

    incorrect_seen = [prev_mat for prev_mat in seen_materials if prev_mat[2]==1]    
    incorrect_seen = [abs(prev_mat[1]-dif_material) for prev_mat in incorrect_seen]
    
    if len(incorrect_seen) == 0:
        return 0
    
    return sum(incorrect_seen)/len(incorrect_seen)

def gap(dif_material, seen_materials, window_l = 3):
    incorrect_seen = [prev_mat for prev_mat in seen_materials if prev_mat[2]==0]
    incorrect_seen = incorrect_seen[::-1][:window_l]
    
    incorrect_seen = [abs(prev_mat[1]-dif_material) for prev_mat in incorrect_seen]
    
    if len(incorrect_seen) == 0:
        return 0
    
    return sum(incorrect_seen)/len(incorrect_seen)

def get_sets(data, skill_learner, seen_materials, sim_threshold=0.2, windows_l=3):
    diff2apt = {}
    diff2exp = {}
    diff2gap = {}
    
    for diff in data.difficulty.unique():
        if diff > skill_learner:
            diff2apt[diff] = apt(diff,skill_learner)
            diff2exp[diff] = exPerf(diff, seen_materials, sim_threshold)
            diff2gap[diff] = gap(diff, seen_materials, windows_l)
            
    return diff2apt, diff2exp, diff2gap


#Random
def get_random_material(df_materials, skill_learner, k = 3):
    materials = df_materials[df_materials.difficulty > skill_learner].sample(frac=1).head(k)
    materials = materials.sort_values('difficulty')
    return materials

def get_alternate_random_material(df_materials, skill_learner, alternate, letter2values, ite, k = 3):
    ite1 = ite%len(alternate)
    min_diff, max_diff = letter2values[ alternate[ite1] ]
    materials = df_materials[(df_materials.difficulty >= min_diff) & (df_materials.difficulty < max_diff)]
    materials = materials[materials.difficulty > skill_learner].sample(frac=1).head(k)
    materials = materials.sort_values('difficulty')
    return materials

def get_random_on_difficulties_material(df_materials, skill_learner, k = 3):
    materials = df_materials[df_materials.difficulty > skill_learner]#.sample(frac=1).head(k)
    difficulties_list = list(materials.difficulty.unique())

    mat = pd.DataFrame(columns = materials.columns)

    for i in range(k):
        one_diff = choice(difficulties_list)
        one_test = materials[materials.difficulty == one_diff].sample(frac=1).head(1)
        mat = mat.append(one_test)

    mat = mat.sort_values('difficulty')
    return mat


# Multi-objective
def get_neighboors(set_tests, diff_hier, skill_learner):
    liste_neighbours = []
    
    for i in range(len(set_tests)):
        for next_dif in diff_hier[ set_tests[i] ]:
            if next_dif > skill_learner:
                test_out = list(set_tests)
                test_out[i] = next_dif

                if test_out != set_tests :
                    liste_neighbours.append(test_out)
            
    return liste_neighbours

def dominated_2dim_apt_exp(elem, list_elem):
    for e in list_elem:
        if e[0] >= elem[0] and e[1] < elem[1]:
            return True
    return False

def dominated_2dim_exp_gap(elem, list_elem):
    for e in list_elem:
        if e[0] < elem[0] and e[1] < elem[1]:
            return True
    return False

def dominated_2dim_apt_gap(elem, list_elem):
    for e in list_elem:
        if e[0] >= elem[0] and e[1] < elem[1]:
            return True
    return False

def dominated_3_dim(elem, list_elem):
    for e in list_elem:
        if e[0] >= elem[0] and e[1] < elem[1] and e[2] < elem[2]:
            return True
    return False


# 3 Objectives
def hill_climbing_3_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests):
    max_cpt = 300
    cpt = 0
    
    #Get the first random solution
    start_sol = get_random_material(df_materials, skill_learner, k)
    tests_set = list(start_sol.difficulty)
    
    #Get Apt and ExPerf of the first Solution
    sol_apt, sol_exp = df_difficulties[df_difficulties.difficulty.isin(tests_set)][['apt','exp']].mean()
    
    while cpt < max_cpt:
        #Get the neighboors
        neighboors = get_neighboors(tests_set, diff_hier, skill_learner)
    
        #Remove neighboors that are dominated by the solution and keep the ones that are not
        apt_exp = []
        to_rem = []
        for nei in neighboors:
            mean_apt, mean_exp = df_difficulties[df_difficulties.difficulty.isin(nei)][['apt','exp']].mean()
            
            if sol_apt < mean_apt or sol_exp >= mean_exp:
                apt_exp.append((mean_apt,mean_exp, mean_apt*mean_exp))
            else:
                to_rem.append(nei)
        
        for nei in to_rem:
            neighboors.remove(nei)
            
        #Remove all dominated solutions
        to_rem_2 = []
        for idx,nei in enumerate(neighboors):
            if dominated_2dim_apt_exp(apt_exp[idx], apt_exp[:idx]+apt_exp[idx:]) == True:
                to_rem_2.append(idx)
        
        for idx in to_rem_2[::-1]:
            del neighboors[idx]
            del apt_exp[idx]
                            
        cpt += 1
        #Get the next solution
        if len(apt_exp) == 0:
            # The solution dominate all the neighboors            
            items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
            start_sol = pd.DataFrame()
            start_sol['item'] = items
            start_sol['difficulty'] = tests_set
            start_sol = start_sol.merge(df_difficulties, on='difficulty')
            return start_sol
        else:
            if sol_apt*sol_exp > max([i[2] for i in apt_exp]):
                #The solution has the best apt*exPerf
                items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
                start_sol = pd.DataFrame()
                start_sol['item'] = items
                start_sol['difficulty'] = tests_set
                start_sol = start_sol.merge(df_difficulties, on='difficulty')
                return start_sol
            else:
                idx = np.random.randint(0,len(neighboors))
                tests_set = neighboors[idx]
                sol_apt, sol_exp = apt_exp[idx][0],apt_exp[idx][1]
    
    items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
    start_sol = pd.DataFrame()
    start_sol['item'] = items
    start_sol['difficulty'] = tests_set
    start_sol = start_sol.merge(df_difficulties, on='difficulty').sort_values('difficulty')
    return start_sol

def get_mo_3_materials(df_materials, df_difficulties, skill_learner, diff2tests, diff_hier, times = 5, k = 3):
    pre_results = []
    
    for tim in range(times):
        sol = hill_climbing_3_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests)
        pre_results.append(sol)
    
    results = [tuple(df[['apt','exp','gap']].mean()) for df in pre_results]
    to_rem = []
    
    for idx,sol in enumerate(results):
        if dominated_3_dim(sol,results[:idx]+results[idx:]) == True:
            to_rem.append(idx)
    
    for idx in to_rem[::-1]:
        del pre_results[idx]
        del results[idx]
    
    idx_best = None
    for idx,sol in enumerate(results):
        if idx == 0:
            idx_best = 0
            
        elif sol[2] < results[idx_best][2]:
            idx_best = idx
    
    return pre_results[idx_best]


# 2 Objectives (ExPerf - Gap)
def hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests):
    max_cpt = 300
    cpt = 0
    
    #Get the first random solution
    start_sol = get_random_material(df_materials, skill_learner, k)
    tests_set = list(start_sol.difficulty)
    
    #Get Apt and ExPerf of the first Solution
    sol_exp = df_difficulties[df_difficulties.difficulty.isin(tests_set)].exp.mean()
    
    while cpt < max_cpt:
        #Get the neighboors
        neighboors = get_neighboors(tests_set, diff_hier, skill_learner)
    
        #Remove neighboors that are dominated by the solution and keep the ones that are not
        apt_exp = []
        to_rem = []
        for nei in neighboors:
            mean_exp = df_difficulties[df_difficulties.difficulty.isin(nei)].exp.mean()
            
            if sol_exp >= mean_exp:
                apt_exp.append(mean_exp)
            else:
                to_rem.append(nei)
        
        for nei in to_rem:
            neighboors.remove(nei)
                            
        cpt += 1
        #Get the next solution
        if len(apt_exp) == 0:
            # The solution dominate all the neighboors            
            items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
            start_sol = pd.DataFrame()
            start_sol['item'] = items
            start_sol['difficulty'] = tests_set
            start_sol = start_sol.merge(df_difficulties, on='difficulty')
            return start_sol
        else:
            max_val = max(apt_exp)
            idx = [i for i,j in enumerate(apt_exp) if j == max_val]
            idx = np.random.choice(idx,1)[0]
            tests_set = neighboors[idx]
            sol_exp = apt_exp[idx]
    
    items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
    start_sol = pd.DataFrame()
    start_sol['item'] = items
    start_sol['difficulty'] = tests_set
    start_sol = start_sol.merge(df_difficulties, on='difficulty').sort_values('difficulty')
    return start_sol

def get_mo_exp_gap_materials(df_materials, df_difficulties, skill_learner, diff2tests, diff_hier, times = 5, k = 3):
    pre_results = []
    
    for tim in range(times):
        sol = hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests)
        pre_results.append(sol)
    
    results = [tuple(df[['exp','gap']].mean()) for df in pre_results]
    to_rem = []
    
    for idx,sol in enumerate(results):
        if dominated_2dim_exp_gap(sol,results[:idx]+results[idx:]) == True:
            to_rem.append(idx)
    
    for idx in to_rem[::-1]:
        del pre_results[idx]
        del results[idx]
    
    idx_best = None
    for idx,sol in enumerate(results):
        if idx == 0:
            idx_best = 0
            
        elif sol[1] < results[idx_best][1]:
            idx_best = idx
    
    return pre_results[idx_best]


# 2 Objectives (Apt - Gap)
def hill_climbing_apt_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests):
    max_cpt = 300
    cpt = 0
    
    #Get the first random solution
    start_sol = get_random_material(df_materials, skill_learner, k)
    tests_set = list(start_sol.difficulty)
    
    #Get Apt and ExPerf of the first Solution
    sol_apt = df_difficulties[df_difficulties.difficulty.isin(tests_set)].apt.mean()
    
    while cpt < max_cpt:
        #Get the neighboors
        neighboors = get_neighboors(tests_set, diff_hier, skill_learner)
    
        #Remove neighboors that are dominated by the solution and keep the ones that are not
        apt_exp = []
        to_rem = []
        for nei in neighboors:
            mean_apt = df_difficulties[df_difficulties.difficulty.isin(nei)].apt.mean()
            
            if sol_apt < mean_apt:
                apt_exp.append(mean_apt)
            else:
                to_rem.append(nei)
        
        for nei in to_rem:
            neighboors.remove(nei)
                            
        cpt += 1
        #Get the next solution
        if len(apt_exp) == 0:
            # The solution dominate all the neighboors            
            items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
            start_sol = pd.DataFrame()
            start_sol['item'] = items
            start_sol['difficulty'] = tests_set
            start_sol = start_sol.merge(df_difficulties, on='difficulty')
            return start_sol
        else:
            max_val = max(apt_exp)
            idx = [i for i,j in enumerate(apt_exp) if j == max_val]
            idx = np.random.choice(idx,1)[0]
            tests_set = neighboors[idx]
            sol_apt = apt_exp[idx]
    
    items = [np.random.choice(diff2tests[diff],1)[0] for diff in tests_set]
    start_sol = pd.DataFrame()
    start_sol['item'] = items
    start_sol['difficulty'] = tests_set
    start_sol = start_sol.merge(df_difficulties, on='difficulty').sort_values('difficulty')
    return start_sol

def get_mo_apt_gap_materials(df_materials, df_difficulties, skill_learner, diff2tests, diff_hier, times = 5, k = 3):
    pre_results = []
    
    for tim in range(times):
        sol = hill_climbing_apt_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests)
        pre_results.append(sol)
    
    results = [tuple(df[['apt','gap']].mean()) for df in pre_results]
    to_rem = []
    
    for idx,sol in enumerate(results):
        if dominated_2dim_apt_gap(sol,results[:idx]+results[idx:]) == True:
            to_rem.append(idx)
    
    for idx in to_rem[::-1]:
        del pre_results[idx]
        del results[idx]
    
    idx_best = None
    for idx,sol in enumerate(results):
        if idx == 0:
            idx_best = 0
            
        elif sol[1] < results[idx_best][1]:
            idx_best = idx
    
    return pre_results[idx_best]

# 2 Objectives (Apt - Gap)
def get_mo_apt_exp_materials(df_materials, df_difficulties, skill_learner, diff2tests, diff_hier, times = 5, k = 3):
    pre_results = []
    
    for tim in range(times):
        sol = hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests)
        pre_results.append(sol)
    
    results = [tuple(df[['apt','exp']].mean()) for df in pre_results]
    to_rem = []
    
    for idx,sol in enumerate(results):
        if dominated_2dim_apt_exp(sol,results[:idx]+results[idx:]) == True:
            to_rem.append(idx)
    
    for idx in to_rem[::-1]:
        del pre_results[idx]
        del results[idx]
    
    idx_best = None
    for idx,sol in enumerate(results):
        if idx == 0:
            idx_best = 0
            
        elif sol[1] > results[idx_best][1]:
            idx_best = idx
    
    return pre_results[idx_best]

# 1 Objective
def get_apt_materials(df_materials, df_difficulties, skill_learner, k = 5):
    results = df_materials[df_materials.difficulty > skill_learner].merge(df_difficulties, on='difficulty')
    max_apt = results.apt.max()
    results1 = results[results.apt == max_apt]
    
    if len(results1) >= k :
        return results1.sample(frac=1).head(k).sort_values('difficulty')
    else:
        results = results.sort_values('apt',ascending=False)
        return results.head(k).sort_values('difficulty')

def get_exp_materials(df_materials, df_difficulties, skill_learner, k = 5):
    results = df_materials[df_materials.difficulty > skill_learner].merge(df_difficulties, on='difficulty')
    max_exp = results.exp.min()
    results1 = results[results.exp == max_exp]

    if len(results1) >= k :
        return results1.sample(frac=1).head(k).sort_values('difficulty')
    else:
        results = results.sort_values('exp')
        return results.head(k).sort_values('difficulty')

def get_gap_materials(df_materials, df_difficulties, skill_learner, k = 5):
    results = df_materials[df_materials.difficulty > skill_learner].merge(df_difficulties, on='difficulty')
    min_gap = results.gap.min()
    results1 = results[results.gap == min_gap]
    
    if len(results1) >= k :
        return results1.sample(frac=1).head(k).sort_values('difficulty')
    else:
        results = results.sort_values('gap')
        return results.head(k).sort_values('difficulty')

