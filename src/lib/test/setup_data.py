from pycaret.classification import setup
from pandas import DataFrame, Series


class mainSetup:
    """"
    Docstring:

    """
    def __init__(self, df:DataFrame, target:Series, normalize:bool, normalize_method:str, preprocess:bool, categorical_features:list,
    n_jobs:int, use_gpu:bool, fold:int, ignore_low_variance=True, html=False, silent=True, log_experiment=False, 
    profile=True) -> None:
        self.df = df
        self.targate = target
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.preprocess = preprocess
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.fold = fold
        self.ignore_low_variance = ignore_low_variance
        self.html = html
        self.silent = silent
        self.log_experiment = log_experiment
        self.profile = profile


    def setup_reg(self):
        reg = setup(data=self.df, target=self.target, normalize=self.normalize,normalize_method=self.normalize_method,
                    preprocess=self.preprocess, categorical_features=self.categorical_features, n_jobs=self.n_jobs, 
                    use_gpu=self.use_gpu, fold=self.fold, ignore_low_variance=self.ignore_low_variance, html=self.html, 
                    silent=self.silent, log_experiment=self.log_experiment, profile=self.profile)
        
        return reg



# def setup_reg(df,chosen_target,norm,norm_method,cpu_num,process_data,cat_features,fold_num,gpu):
#     from pycaret.regression import setup
#     reg = setup(
#                 data=df, target=chosen_target, html= False, silent=True, normalize=bool(norm), 
#                 normalize_method = norm_method, n_jobs=cpu_num, preprocess=bool(process_data),
#                 categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
#                 log_experiment=False, profile=True, use_gpu=bool(gpu)
#                 )
#     return reg



# def setup_clf(df,chosen_target,norm,norm_method,cpu_num,process_data,cat_features,fold_num,gpu):
#     from pycaret.classification import setup
#     clf = setup(
#             data=df, target=chosen_target, html= False, silent=True, normalize=bool(norm), 
#             normalize_method = norm_method, n_jobs=cpu_num, preprocess=bool(process_data),
#             categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
#             log_experiment=False, profile=True, use_gpu=bool(gpu)
#             )
#     return clf


# def setup_cluster(df,chosen_target,norm,norm_method,cpu_num,process_data,cat_features,fold_num,gpu):
#     from pycaret.clustering import setup
#     cluster = setup(
#             data=df, target=chosen_target, html= False, silent=True, normalize=bool(norm), 
#             normalize_method = norm_method, n_jobs=cpu_num, preprocess=bool(process_data),
#             categorical_features=cat_features, ignore_low_variance=True, fold=fold_num,
#             log_experiment=False, profile=True, use_gpu=bool(gpu)
#             )
#     return cluster


