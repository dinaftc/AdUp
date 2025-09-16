from http.client import responses
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
def hill_climbing_3_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k):
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
        sol = hill_climbing_3_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k)
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
def hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k):
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
        sol = hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k)
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
def hill_climbing_apt_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k):
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
        sol = hill_climbing_apt_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k)
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
        sol = hill_climbing_exp_dim(df_materials, df_difficulties, skill_learner, diff_hier, diff2tests,k)
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



EXEC_SYSTEM_PROMPT_TEMPLATE = """
You are an optimization executor for mastery learning. 
You must directly choose k tests from the provided candidates to: 
(1) maximize expected performance = similarity to the learner’s successful tests (by difficulty proximity), 
(2) maximize aptitude = (test_difficulty - skill), 
(3) minimize gap = distance to failed tests (by difficulty proximity). 

Hard constraints:
- Select exactly k unique items.
- All selected tests must have difficulty strictly greater than the learner’s current skill.
- Return ONLY valid JSON, no prose.

Scoring guidance (internal):
- For similarity/distance, treat proximity in difficulty as inverse distance: closer to successes -> higher exPerf; farther from failures -> lower gap.
- Normalize each score to [0,1] within the candidate set to balance objectives.
- Aggregate by a weighted sum with neutral weights exPerf=apt=gap=1, but subtract gap (i.e., exPerf + apt - gap). Ties: prefer higher aptitude.

Example (for illustration only):
skill=0.40, S=[0.42,0.45], F=[0.60]; candidates=[(id:101,d:0.46),(102,0.55),(103,0.62)]; k=2
- exPerf higher when candidate difficulty ~ S (0.46 close to S), 
- apt = d - 0.40,
- gap penalizes closeness to F=0.60.
Expected: pick 101 and 102 (balances aptitude without getting too close to failure at 0.60).


Output strictly as JSON following the provided schema.
Do NOT reveal your reasoning. Return final numbers only.
"""
import json
from typing import Callable

def get_executor_materials(
    data: pd.DataFrame,
    df_difficulties: pd.DataFrame,
    skill_user: float,
    diff2tests: dict = None,
    diff_hier: dict = None,
    k: int = 1,
    chat_fn: Callable = None,
    seen_success_fail: dict = None,
    max_candidates: int = 120
) -> pd.DataFrame:
    """
    LLM-executor: build context (instr/know/mem/state/query), ask LLM
    for k test ids. Returns DataFrame with columns ['item','difficulty'].
    """

    assert chat_fn is not None, "Provide chat_fn(messages)->dict that returns {'content': <string>}"

    # --- Candidate pool: strictly harder than current skill ---
    cand = data[(data['difficulty'] > skill_user) & (data['type'] == 'test')][['id','difficulty']].copy()
    cand = cand.sort_values('difficulty').head(max_candidates)  # bound context size

    # --- Learner memory: successes & failures from df_train (passed via seen_success_fail) ---
    # seen_success_fail = {'S': [(item_id, diff), ...], 'F': [(item_id, diff), ...]}
    S = seen_success_fail.get('S', [])
    print(S)
    F = seen_success_fail.get('F', [])
    print(F)

    # --- Build compact knowledge blobs the model can reason over without tools ---
    # Keep only what’s essential to compute: id, difficulty
    candidates = [{'item': int(r.id), 'difficulty': float(r.difficulty)} for r in cand.itertuples(index=False)]
    successes = [{'item': int(i), 'difficulty': float(d)} for (i, d) in S][-60:]  # limit context
    failures  = [{'item': int(i), 'difficulty': float(d)} for (i, d) in F][-60:] # limit context

    # --- Prompt messages ---
    system_msg = EXEC_SYSTEM_PROMPT_TEMPLATE
    user_msg = json.dumps({
        "objectives": {
            "maximize": ["expected_performance", "aptitude"],
            "minimize": ["gap"],
            "constraints": {"k": k, "strictly_harder_than_skill": True}
        },
        "definitions": {
            "expected_performance": "exPerf(l,t) = similarity(t, S) based on difficulty proximity",
            "aptitude": "apt(l,t) = d_t - skill",
            "gap": "gap(l,t) = distance(t, F) based on difficulty proximity"
        },
        "state": {"skill": skill_user},
        "memory": {"successes": successes, "failures": failures},
        "knowledge": {"candidates": candidates},
        "query": "Select the next batch of k test IDs",
        "output_schema": {
            "type": "object",
            "properties": {
                "selection": {"type": "array", "items": {"type": "integer"}},
                "scores": {
                    "type": "array",
                    "items": {"type": "object", "properties": {
                        "item": {"type": "integer"},
                        "exPerf": {"type": "number"},
                        "apt": {"type": "number"},
                        "gap": {"type": "number"}
                    }, "required": ["item","exPerf","apt","gap"]}
                },
                "notes": {"type": "string"}
            },
            "required": ["selection","scores"]
        }
    }, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    resp = chat_fn(messages) 
    print(resp)
    try:
        payload = json.loads(resp["content"])
        picked = payload["selection"][:k]
        print(picked)
    except Exception:
        print('Exception')
        # Fallback: take the k easiest candidates
        picked = [c['item'] for c in candidates[:k]]

    out = data[data.id.isin(picked)][['id','difficulty']].rename(columns={'id':'item'})
    if len(out) < k:
        print('output shorter than k')
        print(len(out))
        print(k)
        # pad deterministically with nearest difficulties
        need = k - len(out)
        extra = [c['item'] for c in candidates if c['item'] not in set(picked)][:need]
        out = pd.concat([out, data[data.id.isin(extra)][['id','difficulty']].rename(columns={'id':'item'})], ignore_index=True)

    return out[['item','difficulty']]



# ---- Orchestrator tool adapters (wrap your existing functions) ----

def _tool_aptitude(skill_user: float, t_difficulty: float) -> float:
    # uses your apt(diff, skill)
    return apt(t_difficulty, skill_user)

def _tool_experf(t_difficulty: float, seen: list, sim_threshold: float = 0.2) -> float:
    # uses your exPerf(diff, seen, sim_threshold)
    return exPerf(t_difficulty, seen, sim_threshold)

def _tool_gap(t_difficulty: float, seen: list, window_l: int = 3) -> float:
    # uses your gap(diff, seen, window_l)
    return gap(t_difficulty, seen, window_l)

def _tool_neighbors(diff2tests: dict, difficulty: float, k_each: int = 3) -> list[int]:
    # fetch up to k_each concrete item ids at this difficulty
    return diff2tests.get(float(difficulty), [])[:k_each]

def orch_tools_spec():
    return [
        {
            "type": "function",
            "function": {
                "name": "sim_expected_performance",
                "description": "Return exPerf for a target difficulty (lower is better).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_difficulty": {"type": "number"},
                        "sim_threshold": {"type": "number", "default": 0.2}
                    },
                    "required": ["item_difficulty"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "aptitude",
                "description": "Return aptitude = difficulty - skill (higher is better).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_user": {"type": "number"},
                        "item_difficulty": {"type": "number"}
                    },
                    "required": ["skill_user","item_difficulty"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "gap",
                "description": "Return gap = distance to recent failures (lower is better).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_difficulty": {"type": "number"},
                        "window_l": {"type": "integer", "default": 3}
                    },
                    "required": ["item_difficulty"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "neighbors",
                "description": "Return concrete item IDs available at a given difficulty.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "difficulty": {"type": "number"},
                        "k_each": {"type": "integer", "default": 3}
                    },
                    "required": ["difficulty"]
                },
            },
        },
    ]


def ORCH_SYSTEM_PROMPT() -> str:
    return (
        "You are an ORCHESTRATOR for mastery-learning item selection.\n"
        "Use the provided tools (function calling) to get all numeric scores; do not compute them yourself.\n\n"
        "Objectives (according to current metric semantics):\n"
        "  • MINIMIZE exPerf (it is average distance to past successes).\n"
        "  • MAXIMIZE aptitude = difficulty - skill.\n"
        "  • MINIMIZE gap (distance to recent failures).\n"
        "Constraints:\n"
        "  • Select exactly k unique tests, each with difficulty strictly > skill.\n"
        "Protocol:\n"
        "  • Call tools as needed to compare candidates.\n"
        "  • When done, return ONLY a JSON object:\n"
        "    {\"type\":\"final_selection\",\"items\":[{\"item\":int,\"difficulty\":float},...],\"rationale\":\"...\"}\n"
        "No extra prose."
    )


import json
from openai import OpenAI

def my_chat_fn_orch(messages, *, tool_functions, model="gpt-4o"):
    """
    messages: list[{"role":"system"|"user"|"assistant"|"tool","content":str, ...}]
    tool_functions: dict[str, callable] mapping function name -> python callable
    """
    client = OpenAI()

    # We loop until the assistant returns content with no tool_calls
    while True:
        resp = client.chat.completions.create(
            model=model,
            # DO NOT pass response_format here; it prevents tool_calls in some versions.
            messages=messages,
            tools=orch_tools_spec(),
            tool_choice="auto",
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        # Always add the assistant message to the transcript
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls})

        if tool_calls:
            # Execute each tool call locally and append role=tool messages
            for tc in tool_calls:
                fn_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                # Dispatch to your Python tool
                if fn_name not in tool_functions:
                    result = {"error": f"unknown tool '{fn_name}'"}
                else:
                    try:
                        if fn_name == "sim_expected_performance":
                            # needs seen_materials injected at call site
                            result = tool_functions[fn_name](**args)
                        elif fn_name == "aptitude":
                            result = tool_functions[fn_name](**args)
                        elif fn_name == "gap":
                            result = tool_functions[fn_name](**args)
                        elif fn_name == "neighbors":
                            result = tool_functions[fn_name](**args)
                        else:
                            result = tool_functions[fn_name](**args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {str(e)}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,   # <-- REQUIRED to satisfy the API
                    "content": json.dumps(result)
                })
            # Continue loop: the next assistant turn will consider tool results
            continue

        # No tool calls → return final assistant message
        return {"content": msg.content or "{}"}


def get_orchestrator_materials(
    data: pd.DataFrame,
    df_difficulties: pd.DataFrame,
    skill_user: float,
    diff2tests: dict,
    diff_hier: dict,
    k: int,
    chat_fn,  # we'll pass my_chat_fn_orch
    seen_success_fail: dict,
    max_candidates: int = 200
) -> pd.DataFrame:

    # Candidate pool (strictly > skill)
    cand = (
        data[(data['type']=='test') & (data['difficulty'] > skill_user)]
        [['id','difficulty']].drop_duplicates()
        .sort_values('difficulty')
        .head(max_candidates)
        .rename(columns={'id':'item'})
    )
    candidates_preview = cand.head(40).to_dict(orient='records')

    # Build seen list matching your metric signatures: (item, difficulty, correct)
    S = seen_success_fail.get('S', [])
    F = seen_success_fail.get('F', [])
    seen_list = []
    seen_list.extend([(int(i), float(d), 1) for (i, d) in S])
    seen_list.extend([(int(i), float(d), 0) for (i, d) in F])

    # System + User messages
    messages = [
        {"role": "system", "content": ORCH_SYSTEM_PROMPT()},
        {"role": "user", "content": json.dumps({
            "task": "select_next_batch",
            "k": k,
            "skill_user": float(skill_user),
            "candidates_preview": candidates_preview,
            "seen_materials": seen_list,
            "output_schema": {
                "type": "object",
                "required": ["type","items"],
                "properties": {
                    "type": {"enum": ["final_selection"]},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["difficulty"],
                            "properties": {
                                "item": {"type": "integer"},
                                "difficulty": {"type": "number"}
                            }
                        }
                    },
                    "rationale": {"type": "string"}
                }
            }
        })}
    ]

    # Provide stateful closures to the tool loop
    def sim_expected_performance(item_difficulty: float, sim_threshold: float = 0.2):
        return _tool_experf(item_difficulty=item_difficulty, seen_materials=seen_list, sim_threshold=sim_threshold)

    def aptitude(skill_user_: float, item_difficulty: float):
        return _tool_aptitude(skill_user=skill_user_, item_difficulty=item_difficulty)

    def gap(item_difficulty: float, window_l: int = 3):
        return _tool_gap(item_difficulty=item_difficulty, seen_materials=seen_list, window_l=window_l)

    def neighbors(difficulty: float, k_each: int = 3):
        return _tool_neighbors(diff2tests=diff2tests, difficulty=difficulty, k_each=k_each)

    tool_functions = {
        "sim_expected_performance": sim_expected_performance,
        "aptitude": aptitude,
        "gap": gap,
        "neighbors": neighbors,
    }

    # Call the function-calling chat loop
    
    resp = my_chat_fn_orch(messages, tool_functions=tool_functions, model="gpt-4o")

    # Parse final selection
    try:
        payload = json.loads(resp["content"])
        if payload.get("type") == "final_selection":
            selected = payload.get("items", [])
        else:
            selected = []
    except Exception:
        selected = []

    # If difficulties only, map to concrete items via neighbors/diff2tests
    df_out = pd.DataFrame(selected)
    if not df_out.empty and "item" not in df_out.columns and "difficulty" in df_out.columns:
        rows = []
        for d in df_out["difficulty"].tolist():
            pool = diff2tests.get(float(d), [])
            if pool:
                rows.append({"item": int(pool[0]), "difficulty": float(d)})
        df_out = pd.DataFrame(rows)

    # Fallbacks / validation
    if df_out.empty:
        tmp = cand.copy()
        tmp["apt"] = tmp["difficulty"].apply(lambda d: apt(d, skill_user))
        df_out = tmp.sort_values("apt", ascending=False).head(k)[["item","difficulty"]]

    df_out = df_out[df_out["difficulty"] > skill_user].drop_duplicates(subset=["item"])
    if len(df_out) < k:
        need = k - len(df_out)
        already = set(df_out["item"])
        padding = cand[~cand["item"].isin(already)].head(need)
        df_out = pd.concat([df_out, padding], ignore_index=True)

    return df_out[["item","difficulty"]].head(k)
