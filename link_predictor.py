
'''

@todo upddate this string

'''

###########
# Imports #
###########

import os
import json
import karateclub
import numpy as np
import networkx as nx
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm
import pytorch_lightning.metrics.functional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data
from collections import OrderedDict
from typing import Tuple, Dict, List, Set
from typing_extensions import Literal

from misc_utilities import *

# @todo check the imports

###########
# Globals #
###########

RESULT_SUMMARY_JSON_FILE_BASENAME = 'result_summary.json'

TRAINING_PORTION = 0.30
VALIDATION_PORTION = 0.10
TESTING_PORTION = 1 - TRAINING_PORTION - VALIDATION_PORTION

BCE_LOSS = nn.BCELoss(reduction='none')

NODE2VEC_MODEL_FILE_BASENAME = 'node2vec.matrix'
EMBEDDING_VISUALIZATION_FILE_BASENAME = 'node2vec.png'
LINK_PREDICTOR_CHECKPOINT_DIR = './checkpoints'

#################
# Visualization #
#################

def visualize_vectors(matrix: np.ndarray, labels: np.ndarray, output_file_location: str, plot_title: str) -> None:
    assert matrix.shape[0] == len(labels)
    matrix_pca = PCA(n_components=2, copy=False).fit_transform(matrix)
    matrix_tsne = TSNE(n_components=2, init='pca').fit_transform(matrix)
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        def add_plot(position: int, matrix_to_plot: np.ndarray, dimensionality_reduction_style: Literal['PCA', 'TSNE']):
            plot = figure.add_subplot(position)
            plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
            plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
            label_to_color_map = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
            label_to_color_map = dict(enumerate(label_to_color_map))
            colors = np.array([label_to_color_map[label] for label in labels])
            plot.scatter(matrix_to_plot[:,0], matrix_to_plot[:,1], c=colors, alpha=0.25)
            plot.set_title(f'{plot_title} ({dimensionality_reduction_style})')
            plot.set_xlabel(f'{dimensionality_reduction_style} Dim 1')
            plot.set_ylabel(f'{dimensionality_reduction_style} Dim 2')
            plot.grid(True)
        add_plot(121, matrix_pca, 'PCA')
        add_plot(122, matrix_tsne, 'TSNE')
        figure.savefig(output_file_location)
    LOGGER.info(f'Visualization for "{plot_title}" saved at {output_file_location}')

###############
# Data Module #
###############

class FBDataset(data.Dataset):
    def __init__(self, positive_edges: np.ndarray, negative_edges: np.ndarray):
        assert len(positive_edges) == len(negative_edges)
        self.positive_edges = positive_edges
        self.negative_edges = negative_edges
   
    def __getitem__(self, index: int):
        edge_is_positive = bool(index < len(self.positive_edges))
        edge = self.positive_edges[index] if edge_is_positive else self.negative_edges[index - len(self.positive_edges)]
        return {
            'edge': torch.tensor(edge, dtype=int),
            'target': torch.tensor(edge_is_positive, dtype=torch.float32)
        }
   
    def __len__(self):
        return len(self.positive_edges)+len(self.negative_edges)

class FBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, positive_edges: np.ndarray, negative_edges: np.ndarray):
        assert positive_edges.shape == negative_edges.shape
        self.batch_size = batch_size
        self.positive_edges = positive_edges
        self.negative_edges = negative_edges
       
    def prepare_data(self) -> None:
        return
   
    def setup(self) -> None:

        edge_indices = list(range(len(self.positive_edges)))
        training_edge_indices, testing_edge_indices = train_test_split(edge_indices, test_size=1-TRAINING_PORTION, random_state=SEED)
        validation_edge_indices, testing_edge_indices = train_test_split(testing_edge_indices, test_size=TESTING_PORTION/(1-TRAINING_PORTION), random_state=SEED)
       
        training_dataset = FBDataset(self.positive_edges[training_edge_indices], self.negative_edges[training_edge_indices])
        validation_dataset = FBDataset(self.positive_edges[validation_edge_indices], self.negative_edges[validation_edge_indices])
        testing_dataset = FBDataset(self.positive_edges[testing_edge_indices], self.negative_edges[testing_edge_indices])

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/408 forces us to use shuffle in training and drop_last pervasively
        self.training_dataloader = data.DataLoader(training_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)
        self.validation_dataloader = data.DataLoader(validation_dataset, batch_size=len(validation_dataset)//4, num_workers=0, shuffle=False, drop_last=True)
        self.testing_dataloader = data.DataLoader(testing_dataset, batch_size=len(testing_dataset)//4, num_workers=0, shuffle=False, drop_last=True)
       
        assert 0 < len(self.training_dataloader.dataset) == len(training_edge_indices) * 2
        assert 0 < len(self.validation_dataloader.dataset) == len(validation_edge_indices) * 2
        assert 0 < len(self.testing_dataloader.dataset) == len(testing_edge_indices) * 2
       
        assert len(self.testing_dataloader) == len(self.validation_dataloader) == 4
       
        assert round((len(self.training_dataloader.dataset) / 2) / (88234 / 2), 2) == TRAINING_PORTION
        assert round((len(self.validation_dataloader.dataset) / 2) / (88234 / 2), 2) == VALIDATION_PORTION
        assert round((len(self.testing_dataloader.dataset) / 2) / (88234 / 2), 2) == TESTING_PORTION
        
        LOGGER.info(f'Data Module Training Portion: {TRAINING_PORTION}')
        LOGGER.info(f'Data Module Validation Portion: {VALIDATION_PORTION}')
        LOGGER.info(f'Data Module Testing Portion: {TESTING_PORTION}')
        
        return
   
    def train_dataloader(self) -> data.DataLoader:
        return self.training_dataloader

    def val_dataloader(self) -> data.DataLoader:
        return self.validation_dataloader

    def test_dataloader(self) -> data.DataLoader:
        return self.testing_dataloader

########################
# Link Predictor Model #
########################

class LinkPredictor(pl.LightningModule):

    hyperparameter_names = (
        'embedding_size',
        # node2vec Hyperparameters
        'p',
        'q',
        'walks_per_node',
        'walk_length',
        'node2vec_epochs',
        'node2vec_learning_rate',
        # Link Predictor Hyperparameters
        'link_predictor_learning_rate',
        'link_predictor_batch_size',
        'link_predictor_gradient_clip_val',
    )
   
    def __init__(self, graph: nx.Graph, embedding_size: int, p: float, q: float, walks_per_node: int, walk_length: int, node2vec_epochs: int, node2vec_learning_rate: float, link_predictor_learning_rate: float, link_predictor_batch_size: int, link_predictor_gradient_clip_val: float):
        super().__init__()
        self.save_hyperparameters(*(self.__class__.hyperparameter_names))
       
        self.logistic_regression_layers = nn.Sequential(
            OrderedDict([
                (f'linear_layer', nn.Linear(self.hparams.embedding_size, 1)),
                (f'activation_layer', nn.Sigmoid()),
            ])
        )
        self._initialize_embeddings(graph)

    def _initialize_embeddings(self, graph: nx.Graph) -> None:
        checkpoint_directory = self.__class__.checkpoint_directory_from_hyperparameters(**{hyperparameter_name: getattr(self.hparams, hyperparameter_name) for hyperparameter_name in self.__class__.hyperparameter_names})
        if not os.path.isdir(checkpoint_directory):
            os.makedirs(checkpoint_directory)
       
        self.saved_noded2vec_model_location = os.path.join(checkpoint_directory, NODE2VEC_MODEL_FILE_BASENAME)
        self.embedding_visualization_location = os.path.join(checkpoint_directory, EMBEDDING_VISUALIZATION_FILE_BASENAME)
        
        if os.path.isfile(self.saved_noded2vec_model_location):
            with open(self.saved_noded2vec_model_location, 'rb') as f:
                embedding_matrix: np.ndarray = np.load(f)
        else:
            trainer = karateclub.Node2Vec(
                walk_number=self.hparams.walks_per_node,
                walk_length=self.hparams.walk_length,
                workers=mp.cpu_count(),
                p=self.hparams.p,
                q=self.hparams.q,
                dimensions=self.hparams.embedding_size,
                epochs=self.hparams.node2vec_epochs,
                learning_rate=self.hparams.node2vec_learning_rate,
            )
            trainer.fit(graph)
            embedding_matrix: np.ndarray = trainer.get_embedding()
            assert tuple(embedding_matrix.shape) == (len(graph.nodes), self.hparams.embedding_size)
            with open(self.saved_noded2vec_model_location, 'wb') as f:
                np.save(f, embedding_matrix)
            visualize_vectors(embedding_matrix, np.zeros(len(graph.nodes)), self.embedding_visualization_location, 'Embedding Visualization via PCA')

        embedding_matrix: torch.Tensor = torch.from_numpy(embedding_matrix)
        self.embedding_layer = nn.Embedding(embedding_matrix.size(0), embedding_matrix.size(1))
        self.embedding_layer.weight = nn.Parameter(embedding_matrix)
        return

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.shape[0]
        assert tuple(batch.shape) == (batch_size, 2)

        embedded_batch = self.embedding_layer(batch)
        assert tuple(embedded_batch.shape) == (batch_size, 2, self.hparams.embedding_size)

        hadamard_product_batch = embedded_batch[:,0,:] * embedded_batch[:,1,:]
        assert tuple(hadamard_product_batch.shape) == (batch_size, self.hparams.embedding_size), f'{tuple(hadamard_product_batch.shape)} != {(batch_size, self.hparams.embedding_size)}'

        prediction_batch = self.logistic_regression_layers(hadamard_product_batch)
        assert tuple(prediction_batch.shape) == (batch_size, 1)
        prediction_batch = prediction_batch.squeeze(1)
        assert len(prediction_batch.shape) == 1, f'len({prediction_batch.shape}) == 1'
        assert only_one(prediction_batch.shape) == batch_size, f'{only_one(prediction_batch.shape)} != {batch_size}'
       
        return prediction_batch
   
    def backward(self, loss: torch.Tensor , _optimizer: torch.optim.Optimizer, _opt_idx: int) -> None:
        del _optimizer, _opt_idx
        loss.backward()
        return
   
    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.link_predictor_learning_rate)
        return {'optimizer': optimizer}

    @property
    def device(self) -> Union[str, torch.device]:
        return only_one({parameter.device for parameter in self.parameters()})

    def _step(self, batch_dict: dict, eval_type: Literal['training', 'validation', 'testing']) -> Dict[str, torch.Tensor]:
        batch = batch_dict['edge'].to(self.device)
        target_predictions = batch_dict['target'].to(self.device)
        batch_size = only_one(target_predictions.shape)
        assert tuple(batch.shape) == (batch_size, 2)
        assert tuple(target_predictions.shape) == (batch_size,)
       
        predictions = self(batch)
        assert only_one(predictions.shape) == batch_size
        bce_loss = BCE_LOSS(predictions, target_predictions)
        batch_results = {'loss': bce_loss, 'predictions': predictions}
       
        assert len(bce_loss.shape) == 1
        self.log(f'{eval_type}_loss', bce_loss.mean())
       
        return batch_results

    def _aggregate_loss(self, batch_parts_outputs: torch.Tensor, eval_type: Literal['training', 'validation', 'testing']) -> torch.Tensor:
        assert len(batch_parts_outputs.shape) == 1
        loss = batch_parts_outputs.mean()
        self.log(f'{eval_type}_loss', loss)
        return loss

    def training_step(self, batch_dict: dict, batch_index: int) -> torch.Tensor:
        del batch_index
        return self._step(batch_dict, 'training')['loss']

    def training_step_end(self, batch_parts_outputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(batch_parts_outputs, torch.Tensor)
        assert len(batch_parts_outputs.shape) == 1
        return self._aggregate_loss(batch_parts_outputs, 'training')

    def validation_step(self, batch_dict: dict, batch_index: int) -> torch.Tensor:
        del batch_index
        return self._step(batch_dict, 'validation')['loss']

    def validation_step_end(self, batch_parts_outputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(batch_parts_outputs, torch.Tensor)
        assert len(batch_parts_outputs.shape) == 1
        return self._aggregate_loss(batch_parts_outputs, 'validation')

    def test_step(self, batch_dict: dict, batch_index: int) -> Dict[str, torch.Tensor]:
        del batch_index
        batch_results = self._step(batch_dict, 'testing')
        batch_results['target'] = batch_dict['target']
        return batch_results
   
    def test_epoch_end(self, batch_parts_outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        assert isinstance(batch_parts_outputs, list)
        assert all(
            isinstance(batch_parts_output, dict)
            and all(isinstance(key, str) for key in batch_parts_output.keys())
            and all(isinstance(value, torch.Tensor) and len(value.shape) == 1 for value in batch_parts_output.values())
            for batch_parts_output in batch_parts_outputs
        )
        test_results_keys = {'loss', 'predictions', 'target'}
        self.test_results = {key: torch.FloatTensor().to(self.device) for key in test_results_keys}
        for batch_parts_output in batch_parts_outputs:
            assert set(batch_parts_output.keys()) == set(self.test_results.keys()) == test_results_keys
            for test_results_key in test_results_keys:
                self.test_results[test_results_key] = torch.cat([self.test_results[test_results_key], batch_parts_output[test_results_key]])
        assert len(set(map(len, self.test_results.values()))) == 1
        return self._aggregate_loss(self.test_results['loss'], 'testing')
   
    class PrintingCallback(pl.Callback):
   
        def __init__(self, checkpoint_callback: pl.callbacks.ModelCheckpoint):
            super().__init__()
            self.checkpoint_callback = checkpoint_callback
       
        def on_init_start(self, trainer: pl.Trainer) -> None:
            LOGGER.info('')
            LOGGER.info('Initializing trainer.')
            LOGGER.info('')
            return
       
        def on_train_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Model: ')
            LOGGER.info(model)
            LOGGER.info('')
            LOGGER.info(f'Training GPUs: {trainer.gpus}')
            for hyperparameter_name in sorted(model.hparams.keys()):
                LOGGER.info(f'{hyperparameter_name}: {model.hparams[hyperparameter_name]:,}')
            LOGGER.info('')
            LOGGER.info('Data:')
            LOGGER.info('')
            LOGGER.info(f'Training Batch Size: {trainer.train_dataloader.batch_size:,}')
            LOGGER.info(f'Validation Batch Size: {only_one(trainer.val_dataloaders).batch_size:,}')
            LOGGER.info('')
            LOGGER.info(f'Training Batch Count: {len(trainer.train_dataloader):,}')
            LOGGER.info(f'Validation Batch Count: {len(only_one(trainer.val_dataloaders)):,}')
            LOGGER.info('')
            LOGGER.info(f'Training Example Count: {len(trainer.train_dataloader)*trainer.train_dataloader.batch_size:,}')
            LOGGER.info(f'Validation Example Count: {len(only_one(trainer.val_dataloaders))*only_one(trainer.val_dataloaders).batch_size:,}')
            LOGGER.info('')
            LOGGER.info('Starting training.')
            LOGGER.info('')
            return
       
        def on_train_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Training complete.')
            LOGGER.info('')
            return
   
        def on_test_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Starting testing.')
            LOGGER.info('')
            LOGGER.info(f'Testing Batch Size: {only_one(trainer.test_dataloaders).batch_size:,}')
            LOGGER.info(f'Testing Example Count: {len(only_one(trainer.test_dataloaders))*only_one(trainer.test_dataloaders).batch_size:,}')
            LOGGER.info(f'Testing Batch Count: {len(only_one(trainer.test_dataloaders)):,}')
            LOGGER.info('')
            return
       
        def on_test_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Testing complete.')
            LOGGER.info('')
            LOGGER.info(f'Best validation model checkpoint saved to {self.checkpoint_callback.best_model_path} .')
            LOGGER.info('')
            return
   
    @staticmethod
    def checkpoint_directory_from_hyperparameters(
            embedding_size: int,
            p: float,
            q: float,
            walks_per_node: int,
            walk_length: int,
            node2vec_epochs: int,
            node2vec_learning_rate: float,
            link_predictor_learning_rate: float,
            link_predictor_batch_size: int,
            link_predictor_gradient_clip_val: float,
    ) -> str:
        checkpoint_directory = os.path.join(
            LINK_PREDICTOR_CHECKPOINT_DIR,
            f'embed_{int(embedding_size)}_' \
            f'p_{p:.5g}_' \
            f'q_{q:.5g}_' \
            f'walks_{int(walks_per_node)}_' \
            f'walk_length_{int(walk_length)}_' \
            f'n2v_epochs_{int(node2vec_epochs)}_' \
            f'n2v_lr_{node2vec_learning_rate:.5g}_' \
            f'link_lr_{link_predictor_learning_rate:.5g}_' \
            f'link_batch_{int(link_predictor_batch_size)}_' \
            f'link_grad_clip_{link_predictor_gradient_clip_val:.5g}'
        )
        return checkpoint_directory

    @classmethod
    def train_model(cls, gpus: List[int], positive_edges: np.ndarray, negative_edges: np.ndarray, **model_initialization_args) -> float:
        
        hyperparameter_dict = {
            hyperparameter_name: hyperparameter_value
            for hyperparameter_name, hyperparameter_value in model_initialization_args.items()
            if hyperparameter_name in cls.hyperparameter_names
        }
        assert set(cls.hyperparameter_names) == set(hyperparameter_dict.keys())

        checkpoint_dir = cls.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:03d}_{validation_loss}'),
            save_top_k=1,
            verbose=False,
            save_last=True,
            monitor='validation_loss',
            mode='min',
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='validation_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='min',
            strict=True,
        )
        
        trainer = pl.Trainer(
            callbacks=[cls.PrintingCallback(checkpoint_callback), early_stop_callback],
            min_epochs=10,
            gradient_clip_val=model_initialization_args['link_predictor_gradient_clip_val'],
            terminate_on_nan=True,
            gpus=gpus,
            distributed_backend='dp',
            deterministic=True,
            # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
            logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='checkpoint_model'),
            default_root_dir=checkpoint_dir,
            checkpoint_callback=checkpoint_callback,
        )
        
        model = cls(**model_initialization_args)
        
        data_module = FBDataModule(hyperparameter_dict['link_predictor_batch_size'], positive_edges, negative_edges)
        data_module.prepare_data()
        data_module.setup()
        
        trainer.fit(model, data_module)
        test_results = only_one(trainer.test(model, datamodule=data_module, verbose=False, ckpt_path=checkpoint_callback.best_model_path))
        best_validation_loss = checkpoint_callback.best_model_score.item()
        
        assert len(data_module.testing_dataloader.dataset) - only_one(set(map(len, model.test_results.values()))) < 4
        assert int(abs(100*(test_results['testing_loss'] - model.test_results['loss'].mean().item()))) in (0, 1, 2)
        
        testing_auroc = pl.metrics.functional.classification.auroc(model.test_results['predictions'], model.test_results['target'])
        testing_correctness_count = torch.sum(model.test_results['target'].int() == model.test_results['predictions'].round().int()).item()
        testing_accuracy = testing_correctness_count / len(model.test_results['predictions'])
        
        LOGGER.info(f'Testing Loss: {test_results["testing_loss"]}')
        LOGGER.info(f'Testing Accuracy: {testing_correctness_count}/{len(model.test_results["predictions"])} ({testing_accuracy*100:.5g}%)')
        LOGGER.info(f'Testing AUROC: {testing_auroc}')
        
        with open(os.path.join(checkpoint_dir, RESULT_SUMMARY_JSON_FILE_BASENAME), 'w') as f:
            result_summary_dict = hyperparameter_dict.copy()
            result_summary_dict['testing_correctness_count'] = testing_correctness_count
            result_summary_dict['testing_accuracy'] = testing_accuracy
            result_summary_dict['testing_loss'] = test_results['testing_loss']
            result_summary_dict['training_portion'] = TRAINING_PORTION
            result_summary_dict['validation_portion'] = VALIDATION_PORTION
            result_summary_dict['testing_portion'] = TESTING_PORTION
            result_summary_dict['noded2vec_model_location'] = model.saved_noded2vec_model_location
            result_summary_dict['embedding_visualization_location'] = model.embedding_visualization_location
            result_summary_dict['best_validation_model_path'] = checkpoint_callback.best_model_path
            result_summary_dict['best_validation_loss'] = best_validation_loss
            result_summary_dict['training_set_size'] = len(data_module.train_dataloader().dataset)
            result_summary_dict['validation_set_size'] = len(data_module.val_dataloader().dataset)
            result_summary_dict['testing_set_size'] = len(data_module.test_dataloader().dataset)
            json.dump(result_summary_dict, f, indent=4)
        
        return best_validation_loss
