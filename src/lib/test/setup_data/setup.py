"""

Function to setup data for each of machine learning algorithm. This function may change
to suit your task. Call this function in main script.

Example:

>>> from utils.setup_data import setup_data
>>> setup_reg(data=df, target=chosen_target)


"""

from pandas import DataFrame, Series


def setup_reg(df: DataFrame, chosen_target: Series, norm: bool, norm_method: str, cpu_num: int, process_data: bool, cat_features: list, fold_num: int, gpu: bool):
    from pycaret.regression import setup
    reg = setup(
                data=df, target=chosen_target, html= False, silent=True, normalize=norm, 
                normalize_method = norm_method, n_jobs=cpu_num, preprocess=process_data,
                categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
                log_experiment=False, profile=True, use_gpu=gpu
                )
    return reg


def setup_clf(df: DataFrame, chosen_target: Series, norm: bool, norm_method: str, cpu_num: int, process_data: bool, cat_features: list, fold_num: int, gpu: bool):
    from pycaret.classification import setup
    clf = setup(
            data=df, target=chosen_target, html= False, silent=True, normalize=bool(norm), 
            normalize_method = str(norm_method), n_jobs=cpu_num, preprocess=bool(process_data),
            categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
            log_experiment=False, profile=True, use_gpu=bool(gpu)
            )
    return clf



# def setup_cluster(df,chosen_target,norm,norm_method,cpu_num,process_data,cat_features,fold_num,gpu):
#     from pycaret.clustering import setup
#     cluster = setup(
#             data=df, target=chosen_target, html= False, silent=True, normalize=bool(norm), 
#             normalize_method = norm_method, n_jobs=cpu_num, preprocess=bool(process_data),
#             categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
#             log_experiment=False, profile=True, use_gpu=bool(gpu)
#             )
#     return cluster

    
