# RGRecSys
RGRecSys is developed using [RecBole library](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) built-in models and covers several models from general and context-aware recommender systems. It has a robustness evaluation module that allows users to easily and uniformly evaluate recommender system robustness. 
## Requirements

- python >= 3.7 
- torch >= 1.7.0
- numpy >= 1.17.2
- scipy == 1.6.0
- hyperopt >= 0.2.4
- pandas >= 1.0.5
- tqdm >= 4.48.2
- scikit_learn >= 0.23.2
- pyyaml >= 5.1.0
- colorlog == 4.7.2
- colorama == 0.4.4
- tensorboard >= 2.5.0

## Usage

1. If you want to use datasets other than *ml-100k*, you need to use RecBole library to get the atomic files. Otherwise, you can skip this step. The atomic files provide a data representation for different recommendation algorithms including .INTER, .USER, and .ITEM. See [this](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) for more information.
2. Download folder *recbole* from v0.2.1 of RecBole library from [here](https://github.com/RUCAIBox/RecBole/archive/refs/tags/v0.2.1.zip) and locate it in the main folder you downloaded from our GitHub (RGRecSys-master).
3. Create a folder named as "saved" in the RGRecSys-master folder.
4. Specify the model, dataset, and desired robustness test in the main function of the RobustnessGymRecSys.py following  the example below:

```python
if __name__ == '__main__':
  for model in [“BPR”]:  #Specify model here
    dataset = "ml-100k"  #Specify dataset here
    base_config_dict = { #Specify selectively loading data here. Keys are the suffix of loaded atomic files, values are the field name list to be loaded
      'load_col': {
        'inter': ['user_id', 'item_id', 'rating', 'timestamp'], 
        'user': ['user_id', 'age', 'gender','occupation'],
        'item': ['item_id', 'release_year', 'class']
      }
    }
    robustness_dict = {  #Specify the robustness test here. This example shows slicing based on user feature
      "slice": {
        "by_feature": {
          "occupation": {"equal": "student"}
        }
      }
    }
    results = train_and_test(
      model=model, 
      dataset=dataset,
      robustness_tests=robustness_dict,
      base_config_dict=base_config_dict, 
      save_model=False
    )
```

Below is more examples of different robustness test formatting:

###### Slice Test Data by Feature

```python
#Ex: A slice of users whose occupation is student
#Format: ”user feature”: {“equal, min, or max”: “value”}
"slice": {
  "by_feature": {
    "occupation": {"equal": "student"}
  }
}
```
###### Slice Test Data by Interaction

```python
#Ex: A slice of users whose number of interactions is more than 50
#Format: ”user”: {“equal, min, or max”: # interactions}
"slice": {
  "by_inter": {
    "user": {"min": 50}
  }
}
```

###### Sparsify the Training Data

```python
#Ex: randomly drop 25% of interactions for users whose number of interactions is more than 10
#Format: ”min_user_inter”: min num of inter for each user, ”fraction_removed”: fraction of interaction to remove
"sparsify": {
  "min_user_inter": 10,
  "fraction_removed": .25
}
```

###### Transform the Test Data - Structured

```python
#Ex: users age will be replaced with a value between 0.8 of their original age to 1.2 of their original age (user with age 10 will have an age value randomly selected from 8-12)
#Format: ”user or item feature”: fraction of current value that will be added or subtracted from the original value
"transform_features": {
  "structured": {
    "age": 0.2,
  }
}
```


###### Transform the Test Data - Random

```python
#Ex: change 40% of user gender value to any other gender value
#Format: ”user or item feature”: fraction to change
"transform_features": {
  "random": {
    "gender": .40,
  }
}
```

###### Transform the Training Interactions - Random Attack

```python
#Ex: 10% of usser interaction are transformed to other values
#Format: ”fraction_transformed”: fraction to transform
"transform_interactions": {
  "fraction_transformed": 0.1
}
```

###### Distribution Shift in the Test Set

```python
#Ex: manipulate test set to contain 50% male and 50% female
#Format: ”user feature”: {proportions of each feature value}
"distribution_shift": {
  "gender": {
    "M": .5,
    "F": .5
  }
}
```



## Cite

If you aim to use RGRecSys for your research or development, please cite the following paper:
```
@inproceedings{10.1145/3488560.3502192,
author = {Ovaisi, Zohreh and Heinecke, Shelby and Li, Jia and Zhang, Yongfeng and Zheleva, Elena and Xiong, Caiming},
title = {RGRecSys: A Toolkit for Robustness Evaluation of Recommender Systems},
url = {https://doi-org.proxy.cc.uic.edu/10.1145/3488560.3502192},
doi = {10.1145/3488560.3502192},
series = {WSDM '22}
}
```
