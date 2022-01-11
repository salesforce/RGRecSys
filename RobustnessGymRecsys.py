from utils.GlobalVars import *
from recbole.config import Config, EvalSetting
from recbole.sampler import Sampler, RepeatableSampler, KGSampler
from recbole.utils import ModelType, init_logger, get_model, get_trainer, init_seed, InputType
from recbole.utils.utils import set_color
from recbole.data.utils import get_data_loader
from recbole.data import save_split_dataloaders
from RobustnessGymDataset import RobustnessGymDataset
from logging import getLogger, shutdown
import importlib
import pprint as pprint
import pickle


def create_dataset(config):
    """
    Initializes RobustnessGymDataset for each recommendation system type in RecBole.
    Args:
        config (Config): Config file indicating MODEL_TYPE and model.

    Returns:
        RobustnessGymDataset instance.
    """
    dataset_module = importlib.import_module('recbole.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        model_type = config['MODEL_TYPE']
        if model_type == ModelType.SEQUENTIAL:
            from recbole.data.dataset import SequentialDataset
            SequentialDataset.__bases__ = (RobustnessGymDataset,)
            return SequentialDataset(config)
        elif model_type == ModelType.KNOWLEDGE:
            from recbole.data.dataset import KnowledgeBasedDataset
            KnowledgeBasedDataset.__bases__ = (RobustnessGymDataset,)
            return KnowledgeBasedDataset(config)
        elif model_type == ModelType.SOCIAL:
            from recbole.data.dataset import SocialDataset
            SocialDataset.__bases__ = (RobustnessGymDataset,)
            return SocialDataset(config)
        elif model_type == ModelType.DECISIONTREE:
            from recbole.data.dataset import DecisionTreeDataset
            DecisionTreeDataset.__bases__ = (RobustnessGymDataset,)
            return DecisionTreeDataset(config)
        else:
            return RobustnessGymDataset(config)


def get_transformed_train(config, train_kwargs, train_dataloader, robustness_testing_datasets):
    """
    Converts training data set created by transformations into dataloader object. Uses same config
    settings as original training data.

    Args:
        train_kwargs (dict): Training dataset config
        train_dataloader (Dataloader): Training dataloader
        config (Config): General config
        robustness_testing_datasets (dict): Modified datasets resulting from robustness tests

    Returns:
        transformed_train (Dataloader)
    """
    transformed_train = None
    if "transformation_train" in robustness_testing_datasets:
        transformation_kwargs = {
            'config': config,
            'dataset': robustness_testing_datasets['transformation_train'],
            'batch_size': config['train_batch_size'],
            'dl_format': config['MODEL_INPUT_TYPE'],
            'shuffle': True,
        }
        try:
            transformation_kwargs['sampler'] = train_kwargs['sampler']
            transformation_kwargs['neg_sample_args'] = train_kwargs['neg_sample_args']
            transformed_train = train_dataloader(**transformation_kwargs)
        except:
            transformed_train = train_dataloader(**transformation_kwargs)

    return transformed_train


def get_sparsity_train(config, train_kwargs, train_dataloader, robustness_testing_datasets):
    """
    Converts training data set created by sparsity into dataloader object. Uses same config
    settings as original training data.

    Args:
        train_kwargs (dict): Training dataset config
        train_dataloader (Dataloader): Training dataloader
        config (Config): General config
        robustness_testing_datasets (dict): Modified datasets resulting from robustness tests

    Returns:
        sparsity_train (Dataloader)

    """
    sparsity_train = None
    if "sparsity" in robustness_testing_datasets:
        sparsity_kwargs = {
            'config': config,
            'dataset': robustness_testing_datasets['sparsity'],
            'batch_size': config['train_batch_size'],
            'dl_format': config['MODEL_INPUT_TYPE'],
            'shuffle': True,
        }
        try:
            sparsity_kwargs['sampler'] = train_kwargs['sampler']
            sparsity_kwargs['neg_sample_args'] = train_kwargs['neg_sample_args']
            sparsity_train = train_dataloader(**sparsity_kwargs)
        except:
            sparsity_train = train_dataloader(**sparsity_kwargs)

    return sparsity_train


def get_distributional_slice_test(eval_kwargs, test_kwargs, test_dataloader, robustness_testing_datasets):
    """

    Args:
        test_dataloader:
        test_kwargs:
        eval_kwargs (dict):
        test_dataloader (Dataloader):
        robustness_testing_datasets (dict):

    Returns:

    """
    slice_test = None
    if 'distributional_slice' in robustness_testing_datasets:
        slice_kwargs = {'dataset': robustness_testing_datasets['distributional_slice']}
        if 'sampler' in test_kwargs:
            slice_kwargs['sampler'] = test_kwargs['sampler']
        slice_kwargs.update(eval_kwargs)
        slice_test = test_dataloader(**slice_kwargs)

    return slice_test


def get_slice_test(eval_kwargs, test_kwargs, test_dataloader, robustness_testing_datasets):
    """

    Args:
        test_dataloader:
        test_kwargs:
        eval_kwargs (dict):
        test_dataloader (Dataloader):
        robustness_testing_datasets (dict):

    Returns:

    """
    slice_test = None
    if 'slice' in robustness_testing_datasets:
        slice_kwargs = {'dataset': robustness_testing_datasets['slice']}
        if 'sampler' in test_kwargs:
            slice_kwargs['sampler'] = test_kwargs['sampler']
        slice_kwargs.update(eval_kwargs)
        slice_test = test_dataloader(**slice_kwargs)

    return slice_test


def get_transformation_test(eval_kwargs, test_kwargs, test_dataloader, robustness_testing_datasets):
    """

    Args:
        test_dataloader:
        test_kwargs:
        eval_kwargs (dict):
        test_dataloader (Dataloader):
        robustness_testing_datasets (dict):

    Returns:

    """
    transformation_test = None
    if 'transformation' in robustness_testing_datasets:
        transformation_kwargs = {'dataset': robustness_testing_datasets['transformation']}
        if 'sampler' in test_kwargs:
            transformation_kwargs['sampler'] = test_kwargs['sampler']
        transformation_kwargs.update(eval_kwargs)
        transformation_test = test_dataloader(**transformation_kwargs)

    return transformation_test


def data_preparation(config, dataset, save=False):
    """
    Builds datasets, including datasets built by applying robustness tests, configures train, validation, test
    sets, converts to tensors. Overloads RecBole data_preparation - we include the preparation of the robustness test
    train/test/valid sets here.

    Args:
        config (Config):
        dataset (RobustnessGymDataset):
        save (bool):

    Returns:

    """
    model_type = config['MODEL_TYPE']
    model = config['model']
    es = EvalSetting(config)

    original_datasets, robustness_testing_datasets = dataset.build(es)
    train_dataset, valid_dataset, test_dataset = original_datasets
    phases = ['train', 'valid', 'test']
    sampler = None
    logger = getLogger()
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = es.neg_sample_args

    # Training
    train_kwargs = {
        'config': config,
        'dataset': train_dataset,
        'batch_size': config['train_batch_size'],
        'dl_format': config['MODEL_INPUT_TYPE'],
        'shuffle': True,
    }

    if train_neg_sample_args['strategy'] != 'none':
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'`training_neg_sample_num` should be 0 '
                f'if inter_feat have label_field [{dataset.label_field}].'
            )
        if model_type != ModelType.SEQUENTIAL:
            sampler = Sampler(phases, original_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        if model not in ["MultiVAE", "MultiDAE", "MacridVAE", "CDAE", "ENMF", "RaCT", "RecVAE"]:
            train_kwargs['sampler'] = sampler.set_phase('train')
            train_kwargs['neg_sample_args'] = train_neg_sample_args
        if model_type == ModelType.KNOWLEDGE:
            kg_sampler = KGSampler(dataset, train_neg_sample_args['distribution'])
            train_kwargs['kg_sampler'] = kg_sampler

    dataloader = get_data_loader('train', config, train_neg_sample_args)
    logger.info(
        set_color('Build', 'pink') + set_color(f' [{dataloader.__name__}]', 'yellow') + ' for ' +
        set_color('[train]', 'yellow') + ' with format ' + set_color(f'[{train_kwargs["dl_format"]}]', 'yellow')
    )
    if train_neg_sample_args['strategy'] != 'none':
        logger.info(
            set_color('[train]', 'pink') + set_color(' Negative Sampling', 'blue') + f': {train_neg_sample_args}'
        )
    else:
        logger.info(set_color('[train]', 'pink') + set_color(' No Negative Sampling', 'yellow'))
    logger.info(
        set_color('[train]', 'pink') + set_color(' batch_size', 'cyan') + ' = ' +
        set_color(f'[{train_kwargs["batch_size"]}]', 'yellow') + ', ' + set_color('shuffle', 'cyan') + ' = ' +
        set_color(f'[{train_kwargs["shuffle"]}]\n', 'yellow')
    )

    train_data = dataloader(**train_kwargs)
    transformed_train = get_transformed_train(config, train_kwargs, dataloader, robustness_testing_datasets)
    sparsity_train = get_sparsity_train(config, train_kwargs, dataloader, robustness_testing_datasets)

    # Evaluation
    eval_kwargs = {
        'config': config,
        'batch_size': config['eval_batch_size'],
        'dl_format': InputType.POINTWISE,
        'shuffle': False,
    }
    valid_kwargs = {'dataset': valid_dataset}
    test_kwargs = {'dataset': test_dataset}

    if eval_neg_sample_args['strategy'] != 'none':
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'It can not validate with `{es.es_str[1]}` '
                f'when inter_feat have label_field [{dataset.label_field}].'
            )
        if sampler is None:
            if model_type != ModelType.SEQUENTIAL:
                sampler = Sampler(phases, original_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        eval_kwargs['neg_sample_args'] = eval_neg_sample_args
        valid_kwargs['sampler'] = sampler.set_phase('valid')
        test_kwargs['sampler'] = sampler.set_phase('test')

    valid_kwargs.update(eval_kwargs)
    test_kwargs.update(eval_kwargs)

    dataloader = get_data_loader('evaluation', config, eval_neg_sample_args)
    logger.info(
        set_color('Build', 'pink') + set_color(f' [{dataloader.__name__}]', 'yellow') + ' for ' +
        set_color('[evaluation]', 'yellow') + ' with format ' + set_color(f'[{eval_kwargs["dl_format"]}]', 'yellow')
    )
    logger.info(es)
    logger.info(
        set_color('[evaluation]', 'pink') + set_color(' batch_size', 'cyan') + ' = ' +
        set_color(f'[{eval_kwargs["batch_size"]}]', 'yellow') + ', ' + set_color('shuffle', 'cyan') + ' = ' +
        set_color(f'[{eval_kwargs["shuffle"]}]\n', 'yellow')
    )

    valid_data = dataloader(**valid_kwargs)
    test_data = dataloader(**test_kwargs)

    transformed_test = None
    if 'transformation_test' in robustness_testing_datasets:
        transformed_test_kwargs = test_kwargs
        transformed_test_kwargs['dataset'] = robustness_testing_datasets['transformation_test']
        transformed_test = dataloader(**transformed_test_kwargs)

    slice_test = get_slice_test(eval_kwargs, test_kwargs, dataloader, robustness_testing_datasets)
    distributional_slice_test = get_distributional_slice_test(eval_kwargs, test_kwargs, dataloader,
                                                              robustness_testing_datasets)

    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    robustness_testing_data = {'slice': slice_test,
                               'distributional_slice': distributional_slice_test,
                               'transformation_train': transformed_train,
                               'transformation_test': transformed_test,
                               'sparsity': sparsity_train}

    return train_data, valid_data, test_data, robustness_testing_data


def get_config_dict(robustness_tests, base_config_dict):
    """
    Combines robustness_test and train_config_dict into a single config_dict.

    Args:
        robustness_tests (dict): robustness test config dict
        base_config_dict (dict): train/data/eval/model/hyperparam config dict

    Returns:
        config_dict (dict): config dict
    """
    config_dict = {}
    if robustness_tests is not None:
        if base_config_dict is not None:
            config_dict = {**robustness_tests, **base_config_dict}
        else:
            config_dict = robustness_tests
    else:
        if base_config_dict is not None:
            config_dict = base_config_dict
    return config_dict


def train_and_test(model, dataset, robustness_tests=None, base_config_dict=None, save_model=True):
    """
    Train a recommendation model and run robustness tests.
    Args:
        model (str): Name of model to be trained.
        dataset (str): Dataset name; must match the dataset's folder name located in 'data_path' path.
        base_config_dict: Configuration dictionary. If no config passed, takes default values.
        save_model (bool): Determines whether or not to externally save the model after training.
        robustness_tests (dict): Configuration dictionary for robustness tests.

    Returns:

    """

    config_dict = get_config_dict(robustness_tests, base_config_dict)
    config = Config(model=model, dataset=dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    logger = getLogger()
    if len(logger.handlers) != 0:
        logger.removeHandler(logger.handlers[1])
    init_logger(config)

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data, robustness_testing_data = data_preparation(config, dataset, save=True)

    for robustness_test in robustness_testing_data:
        if robustness_testing_data[robustness_test] is not None:
            logger.info(set_color('Robustness Test', 'yellow') + f': {robustness_test}')

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=save_model, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=save_model,
                                   show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    test_result_transformation, test_result_sparsity, \
    test_result_slice, test_result_distributional_slice = None, None, None, None

    if robustness_testing_data['slice'] is not None:
        test_result_slice = trainer.evaluate(robustness_testing_data['slice'], load_best_model=save_model,
                                             show_progress=config['show_progress'])
        logger.info(set_color('test result for slice', 'yellow') + f': {test_result_slice}')

    if robustness_testing_data['distributional_slice'] is not None:
        test_result_distributional_slice = trainer.evaluate(robustness_testing_data['distributional_slice'],
                                                            load_best_model=save_model,
                                                            show_progress=config['show_progress'])
        logger.info(set_color('test result for distributional slice', 'yellow') + f': '
                                                                                  f'{test_result_distributional_slice}')

    if robustness_testing_data['transformation_test'] is not None:
        test_result_transformation = trainer.evaluate(robustness_testing_data['transformation_test'],
                                                      load_best_model=save_model,
                                                      show_progress=config['show_progress'])
        logger.info(set_color('test result for transformation on test', 'yellow') + f': {test_result_transformation}')

    if robustness_testing_data['transformation_train'] is not None:
        transformation_model = get_model(config['model'])(config, robustness_testing_data['transformation_train']).to(
            config['device'])
        logger.info(transformation_model)
        transformation_trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, transformation_model)
        best_valid_score_transformation, best_valid_result_transformation = transformation_trainer.fit(
            robustness_testing_data['transformation_train'], valid_data, saved=save_model,
            show_progress=config['show_progress'])
        test_result_transformation = transformation_trainer.evaluate(test_data, load_best_model=save_model,
                                                                     show_progress=config['show_progress'])
        logger.info(
            set_color('best valid for transformed training set', 'yellow') + f': {best_valid_result_transformation}')
        logger.info(set_color('test result for transformed training set', 'yellow') + f': {test_result_transformation}')

    if robustness_testing_data['sparsity'] is not None:
        sparsity_model = get_model(config['model'])(config, robustness_testing_data['sparsity']).to(config['device'])
        logger.info(sparsity_model)
        sparsity_trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, sparsity_model)
        best_valid_score_sparsity, best_valid_result_sparsity = sparsity_trainer.fit(
            robustness_testing_data['sparsity'], valid_data, saved=save_model,
            show_progress=config['show_progress'])
        test_result_sparsity = sparsity_trainer.evaluate(test_data, load_best_model=save_model,
                                                         show_progress=config['show_progress'])
        logger.info(set_color('best valid for sparsified training set', 'yellow') + f': {best_valid_result_sparsity}')
        logger.info(set_color('test result for sparsified training set', 'yellow') + f': {test_result_sparsity}')

    logger.handlers.clear()
    shutdown()
    del logger

    return {
        'test_result': test_result,
        'distributional_test_result': test_result_distributional_slice,
        'transformation_test_result': test_result_transformation,
        'sparsity_test_result': test_result_sparsity,
        'slice_test_result': test_result_slice
    }


def test(model, dataset, model_path, dataloader_path=None, robustness_tests=None, base_config_dict=None):
    """
    Test a pre-trained model from file path. Note that the only robustness test applicable here
    is slicing.
    Args:
        model (str): Name of model.
        dataset (str): Name of dataset.
        model_path (str): Path to saved model.
        robustness_tests (dict): Configuration dictionary for robustness tests.
        base_config_dict (dict): Configuration dictionary for data/model/training/evaluation.

    Returns:

    """
    config_dict = get_config_dict(robustness_tests, base_config_dict)
    config = Config(model=model, dataset=dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    logger = getLogger()
    if len(logger.handlers) != 0:
        logger.removeHandler(logger.handlers[1])
    init_logger(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    if dataloader_path is None:
        train_data, _, test_data, robustness_testing_data = data_preparation(config, dataset, save=False)
    else:
        train_data, valid_data, test_data = pickle.load(open(SAVED_DIR + dataloader_path, "rb"))
        robustness_testing_data = {"slice": None, "transformation": None, "sparsity": None}

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, model_file=model_path,
                                   show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    test_result_slice = None
    if robustness_testing_data['slice'] is not None:
        test_result_slice = trainer.evaluate(robustness_testing_data['slice'], load_best_model=True,
                                             model_file=model_path,
                                             show_progress=config['show_progress'])
        logger.info(set_color('test result for slice', 'yellow') + f': {test_result_slice}')

    return {
        'test_result': test_result,
        'slice_test_result': test_result_slice
    }


if __name__ == '__main__':
    all_results = {}
    for model in ["BPR"]:
        dataset = "ml-100k"
        base_config_dict = {
            'data_path': DATASETS_DIR,
            'show_progress': False,
            'save_dataset': True,
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                         'user': ['user_id', 'age', 'gender', 'occupation'],
                         'item': ['item_id', 'release_year', 'class']}
        }
        # robustness_dict = {
        # uncomment and add robustness test specifications here
        # }
        results = train_and_test(model=model, dataset=dataset, robustness_tests=robustness_dict,
                                 base_config_dict=base_config_dict)
