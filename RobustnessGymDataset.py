"""
/*
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */
"""

from utils.GlobalVars import *
import copy
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from recbole.utils.utils import set_color, ModelType, init_seed
from recbole.data.dataset import Dataset
from recbole.config import Config, EvalSetting
from recbole.utils.enum_type import FeatureType
from collections.abc import Iterable
from collections import Counter
import random
import numpy as np
import torch


class RobustnessGymDataset(Dataset):
    """
    A RobustnessGymDataset is a modified Dataset.
    """

    def __init__(self, config):
        """

        Args:
            config (Config):
        """
        super().__init__(config)

    def _data_filtering(self):
        """
        Filters data by removing nans, removing duplications,
        updating interaction if nans/duplications removed,
        and resetting index.
        """
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_inter_by_user_or_item()
        self._reset_index()

    def copy(self, new_inter_feat):
        """
        Overloaded copy() in RecBole. This deep copies RobustnessGymDataset and sets inter_feat.
        Args:
            new_inter_feat (RobustnessGymDataset):

        Returns:
            nxt (RobustnessGymDataset):
        """
        nxt = copy.deepcopy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def split_by_ratio(self, ratios, group_by=None):
        """
        Overloaded split_by_ratio in RecBole.
        Main difference - we split RobustnessGymDataset instance (instead of
        Dataloader instance) into train, valid, and test.
        Args:
            ratios (list):
            group_by ():

        Returns:

        """
        self.logger.debug(f'split by ratios [{ratios}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].to_numpy())
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

        self._drop_unused_col()
        next_df = [self.inter_feat.iloc[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def leave_one_out(self, group_by, leave_one_num=1):
        """
        Overloaded leave_one_out in RecBole. Main difference - we split RobustnessGymDataset instance
        (instead of Dataloader instance) into train, valid, and test.
        Args:
            group_by:
            leave_one_num:

        Returns:

        """
        self.logger.debug(f'leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}]')
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
        next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num)

        self._drop_unused_col()
        next_df = [self.inter_feat.iloc[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def _transform_by_field_value_random(self):
        """
        Transforms x% of feature/field values by removing the current value and
        replacing with random value selected from set of all possible values.

        Returns:

        """
        transform_percents = self.config['transform_val']
        if transform_percents is None:
            return []

        self.logger.debug(set_color('transform_by_field_value', 'blue') + f': val={transform_percents}')
        for field in transform_percents:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    # gather all possible field values
                    field_values = []
                    for index, row in feat.iterrows():
                        if not isinstance(row[field], Iterable) and row[field] != 0 and row[field] not in field_values:
                            field_values.append(row[field])
                        elif isinstance(row[field], Iterable) and len(row[field]) != 0:
                            for i in row[field]:
                                if i not in field_values:
                                    field_values.append(i)
                    random_indices = random.sample(range(1, len(feat) - 1),
                                                   round(transform_percents[field] * len(feat) - 1))
                    for i in random_indices:
                        field_value_choices = field_values[:]
                        if not isinstance(feat.iloc[i, feat.columns.get_loc(field)], Iterable):
                            # remove current value and replace with another chosen at random
                            field_value_choices.remove(feat.iloc[i, feat.columns.get_loc(field)])
                            feat.iloc[i, feat.columns.get_loc(field)] = random.choice(field_value_choices)
                        elif isinstance(feat.iloc[i, feat.columns.get_loc(field)], Iterable):
                            for j in feat.iloc[i, feat.columns.get_loc(field)]:
                                field_value_choices.remove(j)
                            # remove iterable and replace with ONE randomly chosen value
                            feat.iloc[i, feat.columns.get_loc(field)] = np.array([[random.choice(field_value_choices)]])
        return field_values

    def _transform_by_field_value_structured(self):
        """
        Transforms field/feature in structured manner.

        (1) If feature value is a single value (float, int), then the value is replaced with a value within x% of the
        current value. For example, age = 30, x = 10% --> may be replaced with age = 32.
        (2) If feature value is an iterable (list, numpy array), then x% of the values are dropped.
        For example, genre = [Horror, Drama, Romance], x = 33% --> may be replaced with genre = [Horror, Romance]
        """

        transform_percents = self.config['DropeFraction_or_variance_transform_val']

        if transform_percents is None:
            return []
        self.logger.debug(set_color('_transform_by_field_value', 'blue') + f': val={transform_percents}')

        for field in transform_percents:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    random_indices = random.sample(range(1, len(feat) - 1),
                                                   round(transform_percents[field] * len(feat) - 1))
                    for i in random_indices:
                        if not isinstance(feat.iloc[i, feat.columns.get_loc(field)], Iterable):
                            # replaces current value with random integer within x% of current value
                            random_value = random.randint(
                                round((1 - transform_percents[field]) * feat.iloc[i, feat.columns.get_loc(field)]),
                                round((1 + transform_percents[field]) * feat.iloc[i, feat.columns.get_loc(field)]))
                            feat.iloc[i, feat.columns.get_loc(field)] = random_value
                        elif isinstance(feat.iloc[i, feat.columns.get_loc(field)], Iterable) and len(
                                feat.iloc[i, feat.columns.get_loc(field)]) > 1:
                            # randomly sample x% from iterable/list and remove them
                            dropped_values = random.sample(list(feat.iloc[i, feat.columns.get_loc(field)]),
                                                           round(transform_percents[field] *
                                                                 len(feat.iloc[i, feat.columns.get_loc(field)])))
                            for item in dropped_values:
                                feat.iat[i, feat.columns.get_loc(field)] = np.array(
                                    feat.iloc[i, feat.columns.get_loc(field)][
                                        feat.iloc[i, feat.columns.get_loc(field)] != item])

    def _transform_by_field_value_delete_feat(self):
        """
        Transforms field by "deleting" x% of feature values. Since the feature value cannot be truly deleted,
        we instead remove x% of feature values and replace with the average value of the feature.
        """

        delete_percent = self.config['DeleteFraction_transform_val']
        if delete_percent is None:
            return []

        self.logger.debug(set_color('_transform_by_field_value', 'blue') + f': val={delete_percent}')
        for field in delete_percent:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            value_list = []
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    # compute average value of feature/field
                    for i in range(len(feat)):
                        value_list.append(feat.iloc[i, feat.columns.get_loc(field)])
                    avg_value = np.mean(value_list)

            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    random_indices = random.sample(range(1, len(feat) - 1),
                                                   round(delete_percent[field] * len(feat) - 1))
                    for i in random_indices:
                        if not isinstance(feat.iloc[i, feat.columns.get_loc(field)], Iterable):
                            # replace with average value of feature
                            feat.iloc[i, feat.columns.get_loc(field)] = avg_value

    def _make_data_more_sparse(self):
        """

        Returns:

        """
        val1 = self.config['selected_user_spars_data']
        val2 = self.config['fraction_spars_data']
        user_D = {}
        item_D = {}

        for line in range(len(self.inter_feat)):
            user_id = self.inter_feat.iloc[line]["user_id"]
            item_id = self.inter_feat.iloc[line]["item_id"]

            if user_id not in user_D:
                user_D[user_id] = []
            user_D[user_id].append(item_id)
            if item_id not in item_D:
                item_D[item_id] = []
            item_D[item_id].append(user_id)

        for user_id in user_D:
            if len(user_D[user_id]) > val1:
                selected_item_id = random.sample(user_D[user_id], round(val2 * len(user_D[user_id])))
                for item in selected_item_id:
                    self.inter_feat.drop(self.inter_feat.loc[self.inter_feat['user_id'] == user_id].loc[
                                             self.inter_feat['item_id'] == item].index, inplace=True)

    def _transform_interactions_random(self):
        """

        Returns:

        """
        transform_fraction = self.config['transform_inter']
        if transform_fraction is None:
            return []

        random_rating = 0
        possible_values = [0.0, 1.0]
        random_rows = random.sample(list(self.inter_feat.index), round(transform_fraction * len(self.inter_feat)))
        for index in random_rows:
            if self.config['MODEL_TYPE'] == ModelType.GENERAL or self.config['MODEL_TYPE'] == ModelType.TRADITIONAL:
                transform_col = "rating"
                get_random_rating = True
                while get_random_rating:
                    random_rating = round(random.uniform(possible_values[0], possible_values[1]), 2)
                    if random_rating != self.inter_feat[transform_col].loc[index]:
                        get_random_rating = False
                self.inter_feat[transform_col].loc[index] = random_rating
            if self.config['MODEL_TYPE'] == ModelType.CONTEXT:
                transform_col = "label"
                if self.inter_feat[transform_col].loc[index] == 1.0:
                    self.inter_feat[transform_col].loc[index] = 0.0
                else:
                    self.inter_feat[transform_col].loc[index] = 1.0

    @staticmethod
    def _get_user_or_item_subset(feat_file, field, val_list):
        """

        Args:
            user_feat (Dataframe):
            feature (str):
            val_list (list):

        Returns:

        """
        return {val: list(feat_file[feat_file[field] == val]) for val in val_list}

    def _distributional_slice_old(self):
        """
        Older implementation of distribution shift based on removing prescribed
        proportions of test subpopulations.
        Returns:

        """
        dist_slice = self.config['distribution_shift']
        print(dist_slice)
        if dist_slice is None:
            return []

        for field in dist_slice:
            distribution = dist_slice[field]
            distribution_keys = list(dist_slice[field].keys())
            print(distribution)
            print(distribution_keys)
            print(len(self.inter_feat))
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            if self.field2type[field] not in {FeatureType.TOKEN}:
                raise ValueError(f'Currently only works for Token types.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    user_dict = {}
                    unique_vals = list(feat[field].unique())
                    for tru_val in unique_vals:
                        user_dict[tru_val] = list(feat[feat[field] == tru_val][self.uid_field])
                    for val, proportion in distribution.items():
                        if val != 0.0:
                            tru_val = self.field2token_id[field][val]
                            for index, row in self.inter_feat.iterrows():
                                if row[self.uid_field] in user_dict[tru_val]:
                                    rand_val = random.uniform(0, 1)
                                    if rand_val <= proportion:
                                        self.inter_feat.drop(index, inplace=True)

    def create_distribution(self):
        """

        Returns:

        """
        dist_shift = self.config['distribution_shift']
        if dist_shift is None:
            return []

        for field in dist_shift:
            distribution_dict = dist_shift[field]
            # supports distribution dict of size 2 only
            assert (len(distribution_dict) == 2)
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            if sum(list(distribution_dict.values())) != 1:
                raise ValueError(f'Distribution needs to add up to 1.')
            if self.field2type[field] not in {FeatureType.TOKEN}:
                raise ValueError(f'Currently only works for Token types.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    user_val_dict = {}
                    user_val_counts = {}
                    user_val_original_proportions = {}
                    unique_vals = list(feat[field].unique())
                    for val in unique_vals:
                        user_val_dict[val] = list(feat[feat[field] == val][self.uid_field])
                        user_val_counts[val] = len(
                            [i for i in self.inter_feat[self.uid_field] if i in user_val_dict[val]])
                    for val, proportion in distribution_dict.items():
                        if val != 0.0:
                            token_val = self.field2token_id[field][val]
                            user_val_original_proportions[val] = user_val_counts[token_val] / len(self.inter_feat)
                    no_change_val = 0
                    no_change_quantity = 0
                    for val, proportion in distribution_dict.items():
                        token_val = self.field2token_id[field][val]
                        if proportion >= user_val_original_proportions[val]:
                            no_change_val = val
                            no_change_new_proportion = proportion
                            no_change_quantity = user_val_counts[token_val]
                    num_new_test = int(no_change_quantity / no_change_new_proportion)
                    num_other_class = num_new_test - no_change_quantity
                    for val, proportion in distribution_dict.items():
                        token_val = self.field2token_id[field][val]
                        if val != no_change_val:
                            original_val = user_val_counts[token_val]
                            drop_indices = np.random.choice(
                                self.inter_feat.index[self.inter_feat[self.uid_field].isin(user_val_dict[token_val])],
                                original_val - num_other_class, replace=False)
                            self.inter_feat = self.inter_feat.drop(drop_indices)
                            new_quantity = len(
                                [i for i in self.inter_feat[self.uid_field] if i in user_val_dict[token_val]])

    @staticmethod
    def create_distribution_slice(train, test):
        print("Preparing distributional test slice.")
        train.get_training_distribution_statistics()
        slice_test = copy.deepcopy(test)
        slice_test.create_distribution()
        # slice_test.get_training_distribution_statistics()
        # slice_test._filter_inter_by_user_or_item()
        slice_test._reset_index()
        slice_test._user_item_feat_preparation()
        return slice_test

    def get_training_distribution_statistics(self):
        """

        Returns:

        """
        dist_slice = self.config['distribution_shift']
        if dist_slice is None:
            print("No Training Stats Computed")
            return []

        for field in dist_slice:
            user_dict = {}
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    unique_vals = list(feat[field].unique())
                    for val in unique_vals:
                        user_dict[val] = list(feat[feat[field] == val][self.uid_field])
            dist = {}
            for val in user_dict:
                if val != 0.0:
                    dist[val] = len(self.inter_feat[self.inter_feat[self.uid_field].isin(user_dict[val])])
            print("Training Distribution:")
            for val in user_dict:
                if val != 0.0:
                    print("Val: ", self.field2id_token[field][int(val)], "Percent: ",
                          dist[val] / sum(list(dist.values())))

    def get_attack_statistics(self, train):
        # TODO: add more statistics
        """

        Args:
            train:

        Returns:

        """
        print("Interaction Transformation Robustness Test Summary")

    def get_distribution_shift_statistics(self, train, test):
        print("Distribution Shift Robustness Test Summary")

    def get_transformation_statistics(self, test):
        # TODO: improve printed information
        print("Transformation of Features Robustness Test Summary")
        print("Original Test Size: ", len(test.inter_feat))
        print("Original Test Users: ", len(test.inter_feat[self.uid_field].unique()))
        print("Original Test Features Distribution")

        print("Transformed Test Size: ", len(self.inter_feat))
        print("Transformed Test Users: ", len(self.inter_feat[self.uid_field].unique()))
        print("Transformed Test Features Distribution")

    def get_sparsity_statistics(self, train):
        """

        Args:
            train:

        Returns:

        """
        print("Sparsity Robustness Test Summary")
        print("Original Train Size: ", len(train.inter_feat))
        print("Original Train Users: ", len(train.inter_feat[self.uid_field].unique()))
        print("Sparsified Train Size: ", len(self.inter_feat))
        print("Sparsified Train Users: ", len(self.inter_feat[self.uid_field].unique()))

    @staticmethod
    def create_transformed_test(test):
        """

        Args:
            test:

        Returns:

        """
        print("Preparing test set transformation.")
        transformed_test = copy.deepcopy(test)
        transformed_test.read_transform_features()
        transformed_test._transform_by_field_value_random()
        transformed_test._transform_by_field_value_structured()
        transformed_test._transform_by_field_value_delete_feat()
        transformed_test.get_transformation_statistics(test)
        return transformed_test

    @staticmethod
    def create_transformed_train(train):
        """

        Returns:

        """
        print("Preparing training set transformation.")
        transformed_train = copy.deepcopy(train)
        transformed_train.read_transform_interactions()
        transformed_train._transform_interactions_random()
        transformed_train.get_attack_statistics(train)
        return transformed_train

    def read_transform_interactions(self):
        transform_config = self.config.final_config_dict["transform_interactions"]

        if transform_config is None:
            print("No transformation configs.")
            return None

        if "fraction_transformed" in transform_config:
            self.config.final_config_dict["transform_inter"] = transform_config["fraction_transformed"]
        else:
            print("No transformation percent specified.")
            return None

    def read_sparsify(self):
        """

        Returns:

        """
        sparsify_config = self.config.final_config_dict["sparsify"]

        if sparsify_config is None:
            print("No sparsity configs.")
            return None

        if "min_user_inter" in sparsify_config:
            min_val = sparsify_config["min_user_inter"]
            self.config.final_config_dict['selected_user_spars_data'] = min_val
        else:
            self.config.final_config_dict['selected_user_spars_data'] = 0

        if "fraction_removed" in sparsify_config:
            fraction = sparsify_config["fraction_removed"]
            self.config.final_config_dict["fraction_spars_data"] = fraction
        else:
            print("No sparsity fraction specified.")
            return None

    @staticmethod
    def create_sparse_train(train):
        """

        Args:
            train:

        Returns:

        """
        print("Preparing sparsified training data set.")
        sparse_train = copy.deepcopy(train)
        sparse_train.read_sparsify()
        sparse_train._make_data_more_sparse()
        sparse_train.get_sparsity_statistics(train)
        return sparse_train

    def _filter_by_inter_num(self, train):
        """
        Overloaded RecBole. This version calls adjusted version of _get_illegal_ids below.
        Args:
            train:

        Returns:

        """
        ban_users = self._get_illegal_ids_by_inter_num(dataset=train, field=self.uid_field, feat=self.user_feat,
                                                       max_num=self.config['max_user_inter_num'],
                                                       min_num=self.config['min_user_inter_num'])
        ban_items = self._get_illegal_ids_by_inter_num(dataset=train, field=self.iid_field, feat=self.item_feat,
                                                       max_num=self.config['max_item_inter_num'],
                                                       min_num=self.config['min_item_inter_num'])

        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        if self.user_feat is not None:
            dropped_user = self.user_feat[self.uid_field].isin(ban_users)
            self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

        if self.item_feat is not None:
            dropped_item = self.item_feat[self.iid_field].isin(ban_items)
            self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

        dropped_inter = pd.Series(False, index=self.inter_feat.index)
        if self.uid_field:
            dropped_inter |= self.inter_feat[self.uid_field].isin(ban_users)
        if self.iid_field:
            dropped_inter |= self.inter_feat[self.iid_field].isin(ban_items)
        self.logger.debug('[{}] dropped interactions'.format(len(dropped_inter)))
        self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(self, dataset, field, feat, max_num=None, min_num=None):
        """
        Overloaded from RecBole. This version uses *train* interactions for slicing.
        Args:
            field:
            feat:
            max_num:
            min_num:

        Returns:

        """
        self.logger.debug('\n get_illegal_ids_by_inter_num:\n\t field=[{}], max_num=[{}], min_num=[{}]'.format(
            field, max_num, min_num
        ))

        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = dataset[field].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        if feat is not None:
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(ids), field))
        return ids

    def _drop_by_value(self, val, cmp):
        """
        Overloaded _drop_by_value function from RecBole Dataset base class.
        Here we enable filtering for any field type (not just floats). We also
        enable dropping of categorical features. This function is called by
        _filter_by_field_value() in RecBole.

        Args:
            val (dict):
            cmp (Callable):

        Returns:
            filter_field (list): field names used in comparison.

        """

        if val is None:
            return []

        self.logger.debug(set_color('drop_by_value', 'blue') + f': val={val}')
        filter_field = []
        for field in val:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    if self.field2type[field] == FeatureType.TOKEN_SEQ:
                        raise NotImplementedError
                    if self.field2type[field] == FeatureType.TOKEN:
                        # tokens are mapped to new values by __init__()
                        if isinstance(val[field], str):
                            feat.drop(feat.index[cmp(feat[field].values, self.field2token_id[field][val[field]])],
                                      inplace=True)
                        else:
                            def convert_to_orig_val(x):
                                if int(x) == 0:
                                    return 0.0
                                else:
                                    try:
                                        return float(self.field2id_token[field][int(x)])
                                    except:
                                        return 0.0

                            original_tokens = np.array([convert_to_orig_val(i) for i in feat[field].values])
                            feat.drop(feat.index[cmp(original_tokens, float(val[field]))], inplace=True)
                    if self.field2type[field] in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                        feat.drop(feat.index[cmp(feat[field].values, val[field])], inplace=True)
            filter_field.append(field)
        return filter_field

    def get_slice_statistics(self, test):
        """

        Args:
            slice_test:
            test:

        Returns:

        """
        print("Slice Robustness Test Summary")
        print("Original Test Size: ", len(test.inter_feat))
        print("Original Test Users: ", len(test.inter_feat[self.uid_field].unique()))
        print("Subpopulation Size: ", len(self.inter_feat))
        print("Subpopulation Users: ", len(self.inter_feat[self.uid_field].unique()))

    def create_slice(self, test, train):
        slice_config = self.config.final_config_dict["slice"]
        slice_test = copy.deepcopy(test)
        print("Preparing subpopulation of Test set.")
        if "by_feature" in slice_config:
            slice_test = self.create_slice_by_feature(slice_test)
        if "by_inter" in slice_config:
            slice_test = self.create_slice_by_inter(slice_test, train)
        slice_test._reset_index()
        slice_test._user_item_feat_preparation()
        slice_test.get_slice_statistics(test)
        return slice_test

    def create_slice_by_inter(self, slice_test, train):
        print("Preparing test set slice based on training set interactions.")
        slice_test.read_slice_by_inter()
        slice_test._filter_by_inter_num(train)
        return slice_test

    def read_slice_by_inter(self):
        feature_config = self.config.final_config_dict["slice"]["by_inter"]

        if feature_config is None:
            print("No interaction subset specified.")
            return None

        if "user" in feature_config:
            user_inter = feature_config["user"]
            assert (type(user_inter) == dict)
            if "min" in user_inter:
                min_val = user_inter["min"]
                self.config.final_config_dict["min_user_inter_num"] = min_val
            if "max" in user_inter:
                max_val = user_inter["max"]
                self.config.final_config_dict["max_user_inter_num"] = max_val
        if "item" in feature_config:
            item_inter = feature_config["item"]
            assert (type(item_inter) == dict)
            if "min" in item_inter:
                min_val = item_inter["min"]
                self.config.final_config_dict["min_item_inter_num"] = min_val
            if "max" in item_inter:
                max_val = item_inter["max"]
                self.config.final_config_dict["max_item_inter_num"] = max_val

    def create_slice_by_feature(self, slice_test):
        print("Preparing test set slice based on feature values.")
        slice_test.read_slice_by_feature()
        slice_test._filter_by_field_value()
        slice_test._filter_inter_by_user_or_item()
        return slice_test

    def read_slice_by_feature(self):
        feature_config = self.config.final_config_dict["slice"]["by_feature"]

        if feature_config is None:
            print("No feature values specified.")
            return None

        for field in feature_config:
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    if field not in self.field2type:
                        raise ValueError(f'Field [{field}] not defined in dataset.')
                    slice_specs = feature_config[field]
                    if type(slice_specs) == dict:
                        if "min" in slice_specs:
                            min_dict = {field: slice_specs["min"]}
                            if self.config.final_config_dict["lowest_val"] is None:
                                self.config.final_config_dict["lowest_val"] = min_dict
                            else:
                                self.config.final_config_dict["lowest_val"].update(min_dict)
                        if "max" in slice_specs:
                            max_dict = {field: slice_specs["max"]}
                            if self.config.final_config_dict["highest_val"] is None:
                                self.config.final_config_dict["highest_val"] = max_dict
                            else:
                                self.config.final_config_dict["highest_val"].update(max_dict)
                        if "equal" in slice_specs:
                            equal_dict = {field: slice_specs["equal"]}
                            if self.config.final_config_dict["equal_val"] is None:
                                self.config.final_config_dict["equal_val"] = equal_dict
                            else:
                                self.config.final_config_dict["equal_val"].update(equal_dict)
                    else:
                        print("Incorrect config format.")
                        return None

    def read_transform_features(self):
        feature_config = self.config.final_config_dict["transform_features"]

        if feature_config is None:
            print("No feature transformation specified.")
            return None

        if "structured" in feature_config:
            self.config.final_config_dict['DropeFraction_or_variance_transform_val'] = {}
            for field in feature_config["structured"]:
                percent = feature_config["structured"][field]
                self.config.final_config_dict['DropeFraction_or_variance_transform_val'].update({field: percent})
        elif "random" in feature_config:
            self.config.final_config_dict['transform_val'] = {}
            for field in feature_config["random"]:
                percent = feature_config["random"][field]
                self.config.final_config_dict['transform_val'].update({field: percent})
        else:
            print("Transformation of features incorrectly specified.")
            return None

    def create_robustness_datasets(self, train, valid, test):
        """
        Create the modified datasets needed for robustness tests according to robustness_dict configurations.
        Args:
            train (RobustnessGymDataset):
            valid (RobustnessGymDataset):
            test (RobustnessGymDataset):

        Returns:

        """
        final_config = self.config.final_config_dict
        robustness_testing_datasets = {}

        if "slice" in final_config:
            robustness_testing_datasets["slice"] = self.create_slice(test, train)

        if "sparsify" in final_config:
            robustness_testing_datasets["sparsity"] = self.create_sparse_train(train)

        if "transform_features" in final_config:
            robustness_testing_datasets['transformation_test'] = self.create_transformed_test(test)

        if "transform_interactions" in final_config:
            robustness_testing_datasets['transformation_train'] = self.create_transformed_train(train)

        if "distribution_shift" in final_config:
            robustness_testing_datasets['distributional_slice'] = self.create_distribution_slice(train, test)

        return robustness_testing_datasets

    def build(self, eval_setting):
        """
        Overloads RecBole build. Our version builds train, valid, test
        and modified versions of train, valid, test as needed according to the
        robustness tests requested in the robustness_dict.
        Args:
            eval_setting (EvalSetting):

        Returns:
            original_datasets (list): list containing original train, valid, test datasets
            robustness_testing_datasets (dict): {robustness test name: modified dataset} key value pairs

        """
        if self.benchmark_filename_list is not None:
            raise NotImplementedError()

        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            self.inter_feat = sk_shuffle(self.inter_feat)
            self.inter_feat = self.inter_feat.reset_index(drop=True)
        elif ordering_args['strategy'] == 'by':
            raise NotImplementedError()

        group_field = eval_setting.group_field
        split_args = eval_setting.split_args

        if split_args['strategy'] == 'by_ratio':
            original_datasets = self.split_by_ratio(split_args['ratios'], group_by=group_field)
        elif split_args['strategy'] == 'by_value':
            raise NotImplementedError()
        elif split_args['strategy'] == 'loo':
            original_datasets = self.leave_one_out(group_by=group_field, leave_one_num=split_args['leave_one_num'])
        else:
            original_datasets = self

        train, valid, test = original_datasets
        robustness_testing_datasets = self.create_robustness_datasets(train, valid, test)

        for data in list(robustness_testing_datasets.values()) + original_datasets:
            if data is not None:
                data.inter_feat = data.inter_feat.reset_index(drop=True)
                data._change_feat_format()
                if ordering_args['strategy'] == 'shuffle':
                    torch.manual_seed(self.config['seed'])
                    data.shuffle()
                elif ordering_args['strategy'] == 'by':
                    data.sort(by=ordering_args['field'], ascending=ordering_args['ascending'])

        return original_datasets, robustness_testing_datasets


if __name__ == '__main__':
    config = Config(model="DCN", dataset="ml-100k",
                    config_dict={'distributional_slicing': {'gender': {"M": .9, "F": .1}}})
    init_seed(config['seed'], config['reproducibility'])
    data = RobustnessGymDataset(config)
    datasets, robust_dict = data.build(EvalSetting(config))
    print(robust_dict.keys())
