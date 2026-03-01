import copy
import itertools
from collections import defaultdict

import pandas as pd
import torch
    
# Reduce candidates to required minimum number of stable or flexible attributes.
def reduce_candidates_min_attributes(K, actionable_attributes, stable_items_binding, min_stable_attributes,
                                     flexible_items_binding, min_flexible_attributes):
    # Reduce by min stable and flexible
    number_of_stable_attributes = len(stable_items_binding) - (min_stable_attributes - K)
    if K > min_stable_attributes:
        number_of_flexible_attributes = len(flexible_items_binding) - (
                min_flexible_attributes - actionable_attributes - 1)
    else:
        number_of_flexible_attributes = 0
    reduced_stable_items_binding = {k: stable_items_binding[k] for k in
                                    list(stable_items_binding.keys())[:number_of_stable_attributes]}
    reduced_flexible_items_binding = {k: flexible_items_binding[k] for k in
                                      list(flexible_items_binding.keys())[:number_of_flexible_attributes]}
    return reduced_stable_items_binding, reduced_flexible_items_binding


# Check if itemset is in the stop list.
def in_stop_list(ar_prefix, stop_list):
    if ar_prefix[-2:] in stop_list:
        return True
    if ar_prefix[1:] in stop_list:
        stop_list.append(ar_prefix)
        return True
    return False


# Generate candidates and count their support
# Node derived from stable attribute must simultaneously have sufficient support for the desired and undesired classes in the consequent.
# Group of nodes representing flexible attributes must simultaneously have required min. support for the desired and undesired classes.
def generate_candidates(ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding, 
                        undesired_mask, desired_mask, idx2col, actionable_attributes=0, item=0, 
                        stop_list=[], frames=None, undesired_state=0, desired_state=1, 
                        stop_list_itemset=[], classification_rules=[], verbose=False):

    K = len(itemset_prefix) + 1
    reduced_stable_items_binding, reduced_flexible_items_binding = reduce_candidates_min_attributes(K,
                                                                                                    actionable_attributes,
                                                                                                    stable_items_binding,
                                                                                                    min_stable_attributes,
                                                                                                    flexible_items_binding,
                                                                                                    min_flexible_attributes)

    if undesired_mask is None:
        undesired_mask = torch.ones(frames[undesired_state].shape[0], dtype=torch.bool, device=device)
        desired_mask = torch.ones(frames[desired_state].shape[0], dtype=torch.bool, device=device)

    undesired_frame = frames[undesired_state]
    desired_frame = frames[desired_state]

    stable_candidates = copy.deepcopy(stable_items_binding)
    flexible_candidates = copy.deepcopy(flexible_items_binding)

    new_branches = []

    # --- stable attributes ---
    for attribute, items in reduced_stable_items_binding.items():
        for item in items:
            new_ar_prefix = ar_prefix + (idx2col[item],)
            if in_stop_list(new_ar_prefix, stop_list):
                continue

            branch_undesired_mask = undesired_mask & (undesired_frame[:, item] > 0)
            branch_desired_mask   = desired_mask & (desired_frame[:, item] > 0)

            undesired_support = branch_undesired_mask.sum().item()
            desired_support = branch_desired_mask.sum().item()

            if verbose:
                print('SUPPORT')
                print(itemset_prefix + (idx2col[item],))
                print((undesired_support, desired_support))
                
            if undesired_support < min_undesired_support or desired_support < min_desired_support:
                stable_candidates[attribute].remove(item)
                stop_list.append(new_ar_prefix)
            else:
                new_branches.append({
                    'ar_prefix': new_ar_prefix,
                    'itemset_prefix': new_ar_prefix,
                    'item': item,
                    'undesired_mask': branch_undesired_mask,
                    'desired_mask': branch_desired_mask,
                    'actionable_attributes': 0,
                })

    for attribute, items in reduced_flexible_items_binding.items():

        new_ar_prefix = ar_prefix + (attribute,)
        if in_stop_list(new_ar_prefix, stop_list):
            continue

        undesired_states = []
        desired_states = []
        undesired_count = 0
        desired_count = 0
        for item in items:

            if in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue

            branch_undesired_mask = undesired_mask & (undesired_frame[:, item] > 0)
            branch_desired_mask   = desired_mask & (desired_frame[:, item] > 0)

            undesired_support = branch_undesired_mask.sum().item()
            desired_support = branch_desired_mask.sum().item()

            if verbose:
                print('SUPPORT')
                print(itemset_prefix + (idx2col[item],))
                print((undesired_support, desired_support))

            # is undesired
            if desired_support + undesired_support == 0:
                undesired_conf = 0
            else:
                undesired_conf = undesired_support / (desired_support + undesired_support)

            if undesired_support >= min_undesired_support:
                undesired_count += 1
                if undesired_conf >= min_undesired_confidence:
                    undesired_states.append({'item': item, 'support': undesired_support, 'confidence': undesired_conf})

            # is desired
            if desired_support + undesired_support == 0:
                desired_conf = 0
            else:
                desired_conf = desired_support / (desired_support + undesired_support)

            if desired_support >= min_desired_support:
                desired_count += 1
                if desired_conf >= min_desired_confidence:
                    desired_states.append({'item': item, 'support': desired_support, 'confidence': desired_conf})

            if desired_support < min_desired_support and undesired_support < min_undesired_support:
                flexible_candidates[attribute].remove(item)
                stop_list_itemset.append(itemset_prefix + (item,))

            # --- append new branch ---
            new_branches.append({
                'ar_prefix': new_ar_prefix,
                'itemset_prefix': itemset_prefix + (idx2col[item],),
                'item': item,
                'undesired_mask': branch_undesired_mask,
                'desired_mask': branch_desired_mask,
                'actionable_attributes': actionable_attributes + 1,
            })

        if actionable_attributes == 0 and (undesired_count == 0 or desired_count == 0):
            if attribute in flexible_candidates:
                del flexible_candidates[attribute]
            stop_list.append(new_ar_prefix)

        # --- update classification rules ---
        elif actionable_attributes + 1 >= min_flexible_attributes:
            for undesired_item in undesired_states:
                new_itemset_prefix = itemset_prefix + (idx2col[undesired_item['item']],)
                classification_rules[new_ar_prefix]['undesired'].append({
                    'itemset': new_itemset_prefix,
                    'support': undesired_item['support'],
                    'confidence': undesired_item['confidence'],
                    'target': desired_change_in_target[0]
                })
            for desired_item in desired_states:
                new_itemset_prefix = itemset_prefix + (idx2col[desired_item['item']],)
                classification_rules[new_ar_prefix]['desired'].append({
                    'itemset': new_itemset_prefix,
                    'support': desired_item['support'],
                    'confidence': desired_item['confidence'],
                    'target': desired_change_in_target[1]
                })


    # --- rebuild stable/flexible candidates for new branches ---
    for new_branch in new_branches:
        adding = False
        new_stable = {}
        new_flexible = {}

        for attribute, items in stable_candidates.items():
            for item in items:
                if adding:
                    new_stable.setdefault(attribute, []).append(item)
                if item == new_branch['item']:
                    adding = True

        for attribute, items in flexible_candidates.items():
            for item in items:
                if adding:
                    new_flexible.setdefault(attribute, []).append(item)
                if item == new_branch['item']:
                    adding = True

        new_branch['stable_items_binding'] = new_stable
        new_branch['flexible_items_binding'] = new_flexible

    return new_branches



# Generate action rules from classification rules
def generate_action_rules(classification_rules, action_rules):
    for attribute_prefix, rules in classification_rules.items():
        for desired_rule in rules['desired']:
            for undesired_rule in rules['undesired']:
                action_rules.append({'undesired': undesired_rule, 'desired': desired_rule})


# Prune tree
def prune_tree(K, classification_rules, stop_list):
    for attribute_prefix, rules in classification_rules.items():
        if K == len(attribute_prefix):
            if len(rules['desired']) < 0 or len(rules['undesired']) < 0:
                stop_list.append(attribute_prefix)
                del classification_rules[attribute_prefix]

def get_bindings(col2idx, stable_attributes, flexible_attributes, target):
    stable_items_binding = defaultdict(list)
    flexible_items_binding = defaultdict(list)
    target_items_binding = defaultdict(list)

    for col_name, idx in col2idx.items():
        # stable
        for attribute in stable_attributes:
            if col_name.startswith(attribute + '_<item>_'):
                stable_items_binding[attribute].append(idx)
                break
        else:
            # flexible
            for attribute in flexible_attributes:
                if col_name.startswith(attribute + '_<item>_'):
                    flexible_items_binding[attribute].append(idx)
                    break
            else:
                # target
                if col_name.startswith(target + '_<item>_'):
                    target_items_binding[target].append(idx)
    return stable_items_binding, flexible_items_binding, target_items_binding


# Create default stop list
def get_stop_list(stable_items_binding, flexible_items_binding):
    stop_list = []
    for items in stable_items_binding.values():
        for stop_couple in itertools.product(items, repeat=2):
            stop_list.append(tuple(stop_couple))
    for item in flexible_items_binding.keys():
        stop_list.append(tuple([item, item]))
    return stop_list

def get_split_tables(data_tensor_2d, target_items_binding, target):
    frames = {}
    
    target_indices = target_items_binding[target]
    target_cols = data_tensor_2d[:, target_indices]

    for i, idx in enumerate(target_indices):
        mask = target_cols[:, i]
        frames[idx] = data_tensor_2d[mask, :]

    return frames

def get_dummies(df):
    df = pd.get_dummies(
        df,
        sparse=False,
        columns=df.columns,
        prefix_sep='_<item>_'
    )
    data_np = df.to_numpy(dtype=bool)
    data_tensor_2d = torch.from_numpy(data_np).to(device)

    col2idx = {}
    idx2col = {}
    for i, col in enumerate(df.columns):
        col2idx[col] = i
        idx2col[i] = col

    return data_tensor_2d, col2idx, idx2col

def init_device(use_cuda):  
    if use_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA not available")
    else:
        return torch.device("cpu")

# Action-Apriori main function
def action_apriori(data, stable_attributes, flexible_attributes, target, desired_change_in_target_l,
                   min_stable_attributes_l, min_flexible_attributes_l, min_undesired_support_l,
                   min_undesired_confidence_l, min_desired_support_l, min_desired_confidence_l, use_cuda=False, verbose=False):
    # Global variables
    global min_stable_attributes
    global min_flexible_attributes
    global min_undesired_support
    global min_desired_support
    global min_undesired_confidence
    global min_desired_confidence
    global desired_change_in_target
    global device
    min_stable_attributes = min_stable_attributes_l
    min_flexible_attributes = min_flexible_attributes_l
    min_undesired_support = min_undesired_support_l
    min_desired_support = min_desired_support_l
    min_undesired_confidence = min_undesired_confidence_l
    min_desired_confidence = min_desired_confidence_l
    desired_change_in_target = desired_change_in_target_l
    device = init_device(use_cuda)

    data_tensor_2d, col2idx, idx2col = get_dummies(data)
    stable_items_binding, flexible_items_binding, target_items_binding = get_bindings(col2idx, stable_attributes,
                                                                                      flexible_attributes, target)
    stop_list = get_stop_list(stable_items_binding, flexible_items_binding)
    frames = get_split_tables(data_tensor_2d, target_items_binding, target)
    undesired_state = target + '_<item>_' + str(desired_change_in_target[0])
    desired_state = target + '_<item>_' + str(desired_change_in_target[1])
    desired_state_idx = col2idx[desired_state]
    undesired_state_idx = col2idx[undesired_state]
    action_rules = []
    classification_rules = defaultdict(lambda: {'desired': [], 'undesired': []})
    stop_list_itemset = []

    candidates_queue = [{
        'ar_prefix': tuple(),
        'itemset_prefix': tuple(),
        'stable_items_binding': stable_items_binding,
        'flexible_items_binding': flexible_items_binding,
        'undesired_mask': None,
        'desired_mask': None,
        'actionable_attributes': 0
    }]
    K = 0
    while len(candidates_queue) > 0:
        candidate = candidates_queue.pop(0)
        if len(candidate['ar_prefix']) > K:
            K += 1
            prune_tree(K, classification_rules, stop_list)
        new_candidates = generate_candidates(**candidate, stop_list=stop_list, frames=frames,
                                             undesired_state=undesired_state_idx, desired_state=desired_state_idx,
                                             stop_list_itemset=stop_list_itemset,
                                             classification_rules=classification_rules, verbose=verbose, idx2col=idx2col)
        candidates_queue += new_candidates
    generate_action_rules(classification_rules, action_rules)
    return action_rules


def get_ar_notation(ar_dict, target):
    rule = '['
    for i, item in enumerate(ar_dict['undesired']['itemset']):
        if i > 0:
            rule += ' ∧ '
        rule += '('
        if item == ar_dict['desired']['itemset'][i]:
            val = item.split('_<item>_')
            rule += str(val[0]) + ': ' + str(val[1])
        else:
            val = item.split('_<item>_')
            val_desired = ar_dict['desired']['itemset'][i].split('_<item>_')
            rule += str(val[0]) + ': ' + str(val[1]) + ' → ' + str(val_desired[1])
        rule += ')'
    rule += '] ⇒ [' + str(target) + ': ' + str(ar_dict['undesired']['target']) + ' → ' + str(
        ar_dict['desired']['target']) + ']'
    rule += ', support of undesired part: ' + str(
        ar_dict['undesired']['support']) + ', confidence of undesired part: ' + str(ar_dict['undesired']['confidence'])
    rule += ', support of desired part: ' + str(ar_dict['desired']['support']) + ', confidence of desired part: ' + str(
        ar_dict['desired']['confidence'])
    return rule