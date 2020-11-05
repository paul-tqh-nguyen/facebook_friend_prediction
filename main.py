
'''

@todo update this string

'''

###########
# Imports #
###########

import argparse
import functools
import os
import json
import pickle
import nvgpu
import random
import more_itertools
import joblib
import optuna
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from typing import Dict, Tuple, Set

from misc_utilities import *
from link_predictor import LinkPredictor, RESULT_SUMMARY_JSON_FILE_BASENAME

# @todo make sure these imports are used

###########
# Globals #
###########

GPU_IDS = eager_map(int, nvgpu.available_gpus())

STUDY_NAME = 'study-link-predictor'
DB_URL = 'sqlite:///study-link-predictor.db'

HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION = './docs/hyperparameter_search_results.json'
NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS = 10_000

PREPROCESSED_DATA_FILE_LOCATION = './preprocessed_data.pickle'

###################
# Data Processing #
###################

def process_data() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    reset_random_seed(SEED)
    if os.path.isfile(PREPROCESSED_DATA_FILE_LOCATION):
        LOGGER.info('Loading previously computed processed data.')
        with open(PREPROCESSED_DATA_FILE_LOCATION, 'rb') as f:
            remaining_graph, positive_edges, negative_edges = pickle.load(f)
    else:
        remaining_graph = nx.Graph()
        with open('./data/facebook_combined.txt', 'r') as f:
            lines = f.readlines()
        edges = [tuple(eager_map(int, line.split())) for line in lines]
        remaining_graph.add_edges_from(edges)
        LOGGER.info('Unprocessed graph loaded.')
        LOGGER.info(f'Unprocessed Graph Node Count: {len(remaining_graph.nodes)}')
        LOGGER.info(f'Unprocessed Graph Edge Count: {len(remaining_graph.edges)}')
    
        assert len(remaining_graph.nodes) == 4039
        assert len(remaining_graph.edges) == 88234
        
        nodes = list(remaining_graph.nodes())
        number_of_edges_to_sample = len(remaining_graph.edges) // 2
        assert number_of_edges_to_sample == 44117
        assert set(nodes) == set(range(4039))
        
        positive_edges = set()
        negative_edges = set()

        with manual_tqdm(total=number_of_edges_to_sample, bar_format='{l_bar}{bar:50}{r_bar}') as progress_bar:
            progress_bar.set_description('Sampling negative edges.')
            while len(negative_edges) < number_of_edges_to_sample:
                random_edge = (random.choice(nodes), random.choice(nodes))
                if len(set(random_edge)) == 2:
                    random_edge = tuple(sorted(random_edge))
                    if random_edge not in negative_edges:
                        if random_edge not in remaining_graph:
                            negative_edges.add(random_edge)
                            progress_bar.update(1)

        with manual_tqdm(total=number_of_edges_to_sample, bar_format='{l_bar}{bar:50}{r_bar}') as progress_bar:
            progress_bar.set_description('Sampling positive edges.')
            while len(positive_edges) < number_of_edges_to_sample:
                random_edge = random.choice(edges)
                if random_edge not in positive_edges:
                    remaining_graph.remove_edge(*random_edge)
                    if nx.is_connected(remaining_graph):
                        positive_edges.add(random_edge)
                        progress_bar.update(1)
                    else:
                        remaining_graph.add_edge(*random_edge)
        
        positive_edges = np.array(list(positive_edges))
        negative_edges = np.array(list(negative_edges))
        
        with open(PREPROCESSED_DATA_FILE_LOCATION, 'wb') as f:
            pickle.dump((remaining_graph, positive_edges, negative_edges), f)

        assert len(positive_edges) == len(negative_edges) == number_of_edges_to_sample
    assert nx.is_connected(remaining_graph)
    
    LOGGER.info(f'Remaining Graph Node Count: {len(remaining_graph.nodes)}')
    LOGGER.info(f'Remaining Graph Edge Count: {len(remaining_graph.edges)}')
    LOGGER.info(f'Positive Edge Count: {len(positive_edges)}')
    LOGGER.info(f'Negative Edge Count: {len(negative_edges)}')
    LOGGER.info('Data processing complete.')
    
    return remaining_graph, positive_edges, negative_edges

########################################
# Link Predictor Hyperparameter Search #
########################################

class LinkPredictorHyperParameterSearchObjective:
    def __init__(self, graph: nx.Graph, positive_edges: np.ndarray, negative_edges: np.ndarray, gpu_id_queue: object):
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dynamically
        self.gpu_id_queue = gpu_id_queue
        self.graph = graph
        self.positive_edges = positive_edges
        self.negative_edges = negative_edges

    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters = {
            'embedding_size': int(trial.suggest_int('embedding_size', 100, 500)),
            # node2vec Hyperparameters
            'p': trial.suggest_uniform('p', 0.25, 4),
            'q': trial.suggest_uniform('q', 0.25, 4),
            'walks_per_node': int(trial.suggest_int('walks_per_node', 6, 20)),
            'walk_length': int(trial.suggest_int('walk_length', 6, 20)),
            'node2vec_epochs': int(trial.suggest_int('node2vec_epochs', 10, 1024)),
            'node2vec_learning_rate': trial.suggest_uniform('node2vec_learning_rate', 1e-6, 1e-2),
            # Link Predictor Hyperparameters
            'link_predictor_learning_rate': trial.suggest_uniform('link_predictor_learning_rate', 1e-6, 1e-2),
            'link_predictor_batch_size': int(trial.suggest_int('link_predictor_batch_size', 1, 2048)),
            'link_predictor_gradient_clip_val': int(trial.suggest_int('link_predictor_gradient_clip_val', 1, 1)),
        }
        assert set(hyperparameters.keys()) == set(LinkPredictor.hyperparameter_names)
        return hyperparameters
    
    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get()

        hyperparameters = self.get_trial_hyperparameters(trial)
        checkpoint_dir = LinkPredictor.checkpoint_directory_from_hyperparameters(**hyperparameters)
        LOGGER.info(f'Starting link predictor training for trial {trial.number} via {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = LinkPredictor.train_model(gpus=[gpu_id], graph=self.graph, positive_edges=self.positive_edges, negative_edges=self.negative_edges, **hyperparameters)
        except Exception as exception:
            if self.gpu_id_queue is not None:
                self.gpu_id_queue.put(gpu_id)
            raise exception
        if self.gpu_id_queue is not None:
            self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def get_number_of_link_predictor_hyperparameter_search_trials(study: optuna.Study) -> int:
    df = study.trials_dataframe()
    if len(df) == 0:
        number_of_remaining_trials = NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS
    else:
        number_of_completed_trials = df.state.eq('COMPLETE').sum()
        number_of_remaining_trials = NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS - number_of_completed_trials
    return number_of_remaining_trials

def load_hyperparameter_search_study() -> optuna.Study:
    return optuna.create_study(study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=DB_URL, direction='minimize', load_if_exists=True)

def hyperparameter_search_study_df() -> pd.DataFrame:
    return load_hyperparameter_search_study().trials_dataframe()

def link_predictor_hyperparameter_search(graph: nx.Graph, positive_edges: np.ndarray, negative_edges: np.ndarray) -> None:
    study = load_hyperparameter_search_study()
    number_of_trials = get_number_of_link_predictor_hyperparameter_search_trials(study)
    optimize_kwargs = dict(
        n_trials=number_of_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )
    assert len(GPU_IDS) > 0, "No GPUs available."
    hyperparameter_trials_per_gpu = 1 # max(1, min(16, mp.cpu_count() // len(GPU_IDS)))
    with mp.Manager() as manager:
        gpu_id_queue = manager.Queue()
        more_itertools.consume((gpu_id_queue.put(gpu_id) for gpu_id in (GPU_IDS * hyperparameter_trials_per_gpu)))
        optimize_kwargs['func'] = LinkPredictorHyperParameterSearchObjective(graph, positive_edges, negative_edges, gpu_id_queue)
        optimize_kwargs['n_jobs'] = len(GPU_IDS) * hyperparameter_trials_per_gpu
        with joblib.parallel_backend('multiprocessing', n_jobs=optimize_kwargs['n_jobs']):
            with training_logging_suppressed():
                study.optimize(**optimize_kwargs)
    return

#########################################
# Hyperparameter Search Result Analysis #
#########################################

def analyze_hyperparameter_search_results() -> None:
    df = hyperparameter_search_study_df()
    df = df.loc[df.state=='COMPLETE']
    params_prefix = 'params_'
    assert set(LinkPredictor.hyperparameter_names) == {column_name[len(params_prefix):] for column_name in df.columns if column_name.startswith(params_prefix)}
    result_summary_dicts = []
    for row in df.itertuples():
        hyperparameter_dict = {hyperparameter_name: getattr(row, params_prefix+hyperparameter_name) for hyperparameter_name in LinkPredictor.hyperparameter_names}
        checkpoint_dir = LinkPredictor.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        result_summary_file_location = os.path.join(checkpoint_dir, RESULT_SUMMARY_JSON_FILE_BASENAME)
        with open(result_summary_file_location, 'r') as f:
            result_summary_dict = json.load(f)
            result_summary_dict['duration_seconds'] = row.duration.seconds
        result_summary_dicts.append(result_summary_dict)
    with open(HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION, 'w') as f:
        json.dump(result_summary_dicts, f, indent=4)
    LOGGER.info(f'Hyperparameter result summary saved to {HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION} .')
    return

#################
# Default Model #
#################

def train_default_model(graph: nx.Graph, positive_edges: np.ndarray, negative_edges: np.ndarray) -> None:
    LinkPredictor.train_model(
        gpus=GPU_IDS,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        graph=graph,
        p=1.75,
        q=3.5,
        walks_per_node=12,
        walk_length=18,
        node2vec_epochs=65,
        node2vec_learning_rate=2e-3,
        embedding_size=360,
        link_predictor_learning_rate=2e-4,
        link_predictor_batch_size=4,
        link_predictor_gradient_clip_val=1.0,
    )
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-train-default-model', action='store_true', help='Train the default classifier.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Perform several trials of hyperparameter search for the link predictor.')
    parser.add_argument('-analyze-hyperparameter-search-results', action='store_true', help=f'Analyze completed hyperparameter search trials.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,map(bool,vars(args).values())))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.train_default_model:
        graph, positive_edges, negative_edges = process_data()
        train_default_model(graph, positive_edges, negative_edges)
    elif args.hyperparameter_search:
        graph, positive_edges, negative_edges = process_data()
        link_predictor_hyperparameter_search(graph, positive_edges, negative_edges)
    elif args.analyze_hyperparameter_search_results:
        analyze_hyperparameter_search_results()
    else:
        raise ValueError('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
