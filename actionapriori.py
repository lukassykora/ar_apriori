import itertools
import copy
from collections import defaultdict   
import pandas as pd

# Reduce candidates to required minimum number of stable or flexible attributes.
def reduce_candidates_min_attributes(K, actionable_attributes, stable_items_binding, min_stable_attributes, flexible_items_binding, min_flexible_attributes):
    #Reduce by min stable and flexible
    number_of_stable_attributes = len(stable_items_binding) - (min_stable_attributes - K)
    if K > min_stable_attributes:
        number_of_flexible_attributes = len(flexible_items_binding) - (min_flexible_attributes - actionable_attributes - 1)
    else:
        number_of_flexible_attributes = 0
    reduced_stable_items_binding = {k: stable_items_binding[k] for k in list(stable_items_binding.keys())[:number_of_stable_attributes]}
    reduced_flexible_items_binding = {k: flexible_items_binding[k] for k in list(flexible_items_binding.keys())[:number_of_flexible_attributes]}
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
def generate_candidates(ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding, unwanted_mask, wanted_mask, actionable_attributes=0, item=0, stop_list=[], frames=None, unwanted_state=0, wanted_state=1, stop_list_itemset=[], classification_rules=[], verbose=False):
    K = len(itemset_prefix) + 1
    reduced_stable_items_binding, reduced_flexible_items_binding = reduce_candidates_min_attributes(K, actionable_attributes, stable_items_binding, min_stable_attributes, flexible_items_binding, min_flexible_attributes)
    
    if unwanted_mask is None:
        unwanted_frame = frames[unwanted_state]
        wanted_frame = frames[wanted_state]
    else:
        unwanted_frame = frames[unwanted_state].multiply(unwanted_mask, axis="index")
        wanted_frame = frames[wanted_state].multiply(wanted_mask, axis="index")
    
    stable_candidates = copy.deepcopy(stable_items_binding)
    flexible_candidates = copy.deepcopy(flexible_items_binding)
    
    new_branches = []
    
    for attribute, items in reduced_stable_items_binding.items():
        for item in items:
            
            new_ar_prefix = ar_prefix + (item, )
            if in_stop_list(new_ar_prefix, stop_list):
                continue
            
            unwanted_support = unwanted_frame[item].sum()
            wanted_support = wanted_frame[item].sum()
            
            if verbose:
                print('SUPPORT')
                print(itemset_prefix + (item, ))
                print((unwanted_support, wanted_support))
            
            if unwanted_support < min_unwanted_support or wanted_support < min_wanted_support:
                stable_candidates[attribute].remove(item)
                stop_list.append(new_ar_prefix)
            else:
                new_branches.append({'ar_prefix': new_ar_prefix,
                                   'itemset_prefix': new_ar_prefix,
                                   'item': item,  
                                   'unwanted_mask': unwanted_frame[item],
                                   'wanted_mask': wanted_frame[item],
                                   'actionable_attributes': 0,
                                  })
                 
    for attribute, items in reduced_flexible_items_binding.items():
        
        new_ar_prefix = ar_prefix + (attribute, )
        if in_stop_list(new_ar_prefix, stop_list):
            continue
            
        unwanted_states = []
        wanted_states = []
        unwanted_count = 0
        wanted_count = 0
        for item in items:
            
            if in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue
            
            unwanted_support = unwanted_frame[item].sum()
            wanted_support = wanted_frame[item].sum()
            
            if verbose:
                print('SUPPORT')
                print(itemset_prefix + (item,))
                print((unwanted_support, wanted_support))

            # is unwanted
            if wanted_support + unwanted_support == 0:
                unwanted_conf = 0
            else:
                unwanted_conf = unwanted_support/(wanted_support + unwanted_support)
            if unwanted_support >= min_unwanted_support:
                unwanted_count += 1
                if unwanted_conf >= min_unwanted_confidence:
                    unwanted_states.append({'item': item, 'support': unwanted_support, 'confidence':unwanted_conf})
            # is wanted
            if wanted_support + unwanted_support == 0:
                wanted_conf = 0
            else:
                wanted_conf = wanted_support/(wanted_support + unwanted_support) 
            if wanted_support >= min_wanted_support:
                wanted_count += 1
                if wanted_conf >= min_wanted_confidence:
                    wanted_states.append({'item': item, 'support': wanted_support, 'confidence': wanted_conf})         
            if wanted_support < min_wanted_support and unwanted_support < min_unwanted_support:
                flexible_candidates[attribute].remove(item)
                stop_list_itemset.append(itemset_prefix + (item,))
                
        if actionable_attributes == 0 and (unwanted_count == 0 or wanted_count == 0): # just for first flexible level
            del flexible_candidates[attribute]
            stop_list.append(ar_prefix + (attribute, ))  
        else:
            for item in items: 
                new_branches.append({'ar_prefix': new_ar_prefix,
                                   'itemset_prefix': itemset_prefix + (item,),
                                   'item': item,
                                   'unwanted_mask': unwanted_frame[item],
                                   'wanted_mask': wanted_frame[item],
                                   'actionable_attributes': actionable_attributes + 1,
                                  })
                
            if actionable_attributes + 1 >= min_flexible_attributes:
                for unwanted_item in unwanted_states:
                    new_itemset_prefix = itemset_prefix + (unwanted_item['item'], )
                    classification_rules[new_ar_prefix]['unwanted'].append({
                                         'itemset': new_itemset_prefix,  
                                         'support': unwanted_item['support'],
                                         'confidence': unwanted_item['confidence'],
                                         'target': wanted_change_in_target[0]
                                        })
                for wanted_item in wanted_states:
                    new_itemset_prefix = itemset_prefix + (wanted_item['item'], )
                    classification_rules[new_ar_prefix]['wanted'].append({
                                         'itemset':new_itemset_prefix, 
                                         'support': wanted_item['support'],
                                         'confidence': wanted_item['confidence'],
                                         'target': wanted_change_in_target[1]
                                        })
    
    for new_branch in new_branches:
        adding = False
        new_stable = {}
        new_flexible = {}
        
        for attribute, items in stable_candidates.items():
            for item in items:
                if adding:
                    if attribute not in new_stable:
                        new_stable[attribute] = []
                    new_stable[attribute].append(item)
                if item == new_branch['item']:
                    adding = True
                
                    
                    
        for attribute, items in flexible_candidates.items():
            for item in items:
                if adding:
                    if attribute not in new_flexible:
                        new_flexible[attribute] = []
                    new_flexible[attribute].append(item)
                if item == new_branch['item']:
                    adding = True
                
        new_branch['stable_items_binding'] = new_stable
        new_branch['flexible_items_binding'] = new_flexible
        
    return new_branches

# Generate action rules from classification rules
def generate_action_rules(classification_rules, action_rules):
    for attribute_prefix, rules in classification_rules.items():            
        for wanted_rule in rules['wanted']:
            for unwanted_rule in rules['unwanted']:
                action_rules.append({'unwanted': unwanted_rule, 'wanted': wanted_rule})

# Prune tree
def prune_tree(K, classification_rules, stop_list):
    for attribute_prefix, rules in classification_rules.items():
        if K == len(attribute_prefix):
            if len(rules['wanted']) < 0 or len(rules['unwanted']) < 0:
                stop_list.append(attribute_prefix)    
                del classification_rules[attribute_prefix]

# Get the dictionaries of attributes and their values
def get_bindings(data, stable_attributes, flexible_attributes, target):
    stable_items_binding = defaultdict(lambda: [])
    flexible_items_binding = defaultdict(lambda: [])
    target_items_binding = defaultdict(lambda: [])

    for col in data.columns:
        is_continue = False
        # stable
        for attribute in stable_attributes:
            if col.startswith(attribute+'_<item>_'):
                stable_items_binding[attribute].append(col)
                is_continue = True
                break
        if is_continue is True:
            continue
        # flexible    
        for attribute in flexible_attributes:
            if col.startswith(attribute+'_<item>_'):
                flexible_items_binding[attribute].append(col)
                is_continue = True
                break
        if is_continue is True:
            continue
        # target    
        if col.startswith(target+'_<item>_'):
            target_items_binding[target].append(col) 
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

# Split data to desired and undesired
def get_split_tables(data, target_items_binding, target):
    frames = {}
    for item in target_items_binding[target]:
        mask = data[item]==1
        frames[item] = data[mask]
    return frames


# Action-Apriori main function
def action_apriori(data, stable_attributes, flexible_attributes, target, wanted_change_in_target_l, min_stable_attributes_l , min_flexible_attributes_l, min_unwanted_support_l, min_unwanted_confidence_l, min_wanted_support_l, min_wanted_confidence_l, verbose=False):
    # Global variables
    global min_stable_attributes
    global min_flexible_attributes
    global min_unwanted_support
    global min_wanted_support
    global min_unwanted_confidence
    global min_wanted_confidence
    global wanted_change_in_target
    min_stable_attributes = min_stable_attributes_l
    min_flexible_attributes = min_flexible_attributes_l
    min_unwanted_support = min_unwanted_support_l
    min_wanted_support = min_wanted_support_l
    min_unwanted_confidence = min_unwanted_confidence_l
    min_wanted_confidence = min_wanted_confidence_l
    wanted_change_in_target = wanted_change_in_target_l

    data = pd.get_dummies(data, sparse=False, columns=data.columns, prefix_sep='_<item>_')
    stable_items_binding, flexible_items_binding, target_items_binding = get_bindings(data, stable_attributes, flexible_attributes, target)
    stop_list = get_stop_list(stable_items_binding, flexible_items_binding)
    frames = get_split_tables(data, target_items_binding, target)
    unwanted_state = target + '_<item>_' + str(wanted_change_in_target[0])
    wanted_state = target + '_<item>_' + str(wanted_change_in_target[1])
    action_rules = []
    classification_rules = defaultdict(lambda: {'wanted': [], 'unwanted': []})
    stop_list_itemset = []
    
    candidates_queue = [{
                         'ar_prefix': tuple(),
                         'itemset_prefix':tuple(), 
                         'stable_items_binding': stable_items_binding, 
                         'flexible_items_binding': flexible_items_binding,
                         'unwanted_mask': None,
                         'wanted_mask': None,
                         'actionable_attributes':0
                        }]
    K = 0
    while len(candidates_queue)>0:
        candidate = candidates_queue.pop(0)
        if len(candidate['ar_prefix']) > K:
            K+=1
            prune_tree(K, classification_rules, stop_list)
        new_candidates = generate_candidates(**candidate, stop_list=stop_list, frames=frames, unwanted_state=unwanted_state, wanted_state=wanted_state, stop_list_itemset=stop_list_itemset, classification_rules=classification_rules, verbose=verbose)
        candidates_queue += new_candidates
    generate_action_rules(classification_rules, action_rules)
    return action_rules

def get_ar_notation(ar_dict, target):
    rule = '['
    for i, item  in enumerate(ar_dict['unwanted']['itemset']):
        if i > 0:
            rule += ' ∧ '
        rule += '('
        if item == ar_dict['wanted']['itemset'][i]:
            val = item.split('_<item>_')
            rule += str(val[0]) + ': ' + str(val[1])
        else:
            val = item.split('_<item>_')
            val_wanted = ar_dict['wanted']['itemset'][i].split('_<item>_')
            rule += str(val[0]) + ': ' + str(val[1]) + ' → ' + str(val_wanted[1])
        rule += ')'
    rule += '] ⇒ [' + str(target) + ': ' + str(ar_dict['unwanted']['target']) + ' → ' + str(ar_dict['wanted']['target']) + ']'
    rule += ', support of undesired part: ' + str(ar_dict['unwanted']['support']) + ', confidence of undesired part: ' + str(ar_dict['unwanted']['confidence'])
    rule += ', support of desired part: ' + str(ar_dict['wanted']['support']) + ', confidence of desired part: ' + str(ar_dict['wanted']['confidence']) 
    return rule