# Action-Apriori (Apriori Modified for Action Rules Mining)

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


 ## Installation
The Action-Apriori script needs the following libraries:
- itertools (built-in module in python)
- copy (built-in module in python)
- collections (built-in module in python)
- pandas (1.3.4)

The tested Python version is: 3.9.7

Download the script actionapriori.py and import it. Then the action_apriori function can be called:

```python
import actionapriori
import pandas as pd
# Data
transactions = {'Sex': ['M', 'F', 'M', 'M', 'F', 'M', 'F'], 
                'Age': ['Y', 'Y', 'O', 'Y', 'Y', 'O', 'Y'],
                'Class': [1, 1, 2, 2, 1, 1, 2],
                'Embarked': ['S', 'C', 'S', 'C', 'S', 'C', 'C'],
                'Survived': [1, 1, 0, 0, 1, 1, 0],
               }
data = pd.DataFrame.from_dict(transactions)
# Parameters
stable_attributes = ['Sex','Age']
flexible_attributes = ['Class','Embarked']
target = 'Survived'
wanted_change_in_target = [0, 1]
min_stable_attributes = 2
min_flexible_attributes = 1 #min 1
min_unwanted_support = 1
min_unwanted_confidence = 0.5 #min 0.5
min_wanted_support = 2
min_wanted_confidence = 0.5 #min 0.5
# Action Rules Mining
action_rules = actionapriori.action_apriori(
    data, 
    stable_attributes, 
    flexible_attributes, 
    target, 
    wanted_change_in_target,
    min_stable_attributes , 
    min_flexible_attributes, 
    min_unwanted_support, 
    min_unwanted_confidence, 
    min_wanted_support, 
    min_wanted_confidence, 
    True) #verbose
# Print rules
for action_rule in action_rules:
    print(action_rule)
# Print rules with action rules notation
for action_rule in action_rules:
    print(actionapriori.get_ar_notation(action_rule, target))
```

The output: ation rule with notation:

```python
[(Sex: F) ∧ (Age: Y) ∧ (Class: 2 → 1)] ⇒ [Survived: 0 → 1], support of undesired part: 1, confidence of undesired part: 1.0, support of desired part: 2, confidence of desired part: 1.0
```

## Example

The Example (simplified Titanic data) [action-apriori](https://github.com/lukassykora/ar_apriori/blob/main/Action-Apriori%20Example.ipynb).

## Performance

See [performance](https://github.com/lukassykora/ar_apriori/blob/main/Performance.ipynb) on full Titanic dataset.
