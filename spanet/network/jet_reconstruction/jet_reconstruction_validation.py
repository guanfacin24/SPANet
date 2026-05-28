from typing import Dict, Callable
import warnings
from collections import defaultdict

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.disco_loss import disco_loss


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        self.options = options
        if self.balance_particles:
            self.particle_index_tensor_np = self.particle_index_tensor.cpu().detach().numpy()
            self.particle_weights_tensor_np = self.particle_weights_tensor.cpu().detach().numpy()
        # self.validation_step_metrics_outputs = []

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(
            self,
            jet_predictions,
            particle_scores,
            stacked_targets,
            stacked_masks,
            stacked_weights,
            custom_weights
    ):

        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        weighted_jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target, weight) in enumerate(zip(jet_predictions, stacked_targets[permutation], stacked_weights[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)
                weighted_jet_accuracies[i, j] = np.all(prediction == target, axis=1) * weight

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        weighted_jet_accuracies = weighted_jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        tot_target_weights = (stacked_masks * stacked_weights).sum(0)
        jet_accuracies = jet_accuracies.max(0)
        weighted_jet_accuracies = weighted_jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        has_targets = tot_target_weights > 0
        weighted_avg_jet_accuracy = weighted_jet_accuracies[has_targets] / tot_target_weights[has_targets]
        metrics["validation_average_jet_accuracy"] = np.mean(weighted_avg_jet_accuracy)

        # Compute reconstruction accuracies for all targets as well as accuracy of detection prediction
        particle_names = self.event_info.event_particles.names
        for i, name in enumerate(particle_names):

            # assignment
            sorted_predictions = np.sort(jet_predictions[i], axis = 1)
            sorted_targets = np.sort(stacked_targets[i], axis = 1)
            mask = stacked_masks[i]
            is_correct = np.all(sorted_predictions == sorted_targets, axis = 1)
            accuracy = np.mean((is_correct * custom_weights.numpy())[mask])
            metrics[f"EVENT/{name}_accuracy"] = accuracy

            # detection
            accuracy = np.mean((particle_predictions[i] == stacked_masks[i]) * custom_weights.numpy())
            metrics[f"EVENT/{name}_detection"] = accuracy

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:

        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets, item = batch
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        # get additional variables
        custom_weights = self.validation_dataset.custom_weights[batch.item.detach().cpu().numpy()]
        correlations = self.validation_dataset.correlations[batch.item.detach().cpu().numpy()]

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        stacked_weights = np.zeros((num_targets, batch_size), dtype=float)
        for i, (target, mask, weight) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()
            stacked_weights[i] = weight.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(
            self.compute_metrics(
                jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights, custom_weights
            )
        )

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            metrics[f"REGRESSION/{key}_percent_error"] = percent_error.mean()

            absolute_error = np.abs(delta)
            metrics[f"REGRESSION/{key}_absolute_error"] = absolute_error.mean()

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:

            accuracy = (np.round(classifications[key]) == classification_targets[key])
            accuracy = np.mean(accuracy * custom_weights.numpy())

            roc_auc = sk_metrics.roc_auc_score(
                classification_targets[key], classifications[key], sample_weight = custom_weights
            )

            metrics[f"CLASSIFICATION/{key}_accuracy"] = accuracy
            metrics[f"CLASSIFICATION/{key}_roc-auc"] = roc_auc

            # check whether correlations are passed (they are 1 by default)
            if np.any(correlations.numpy() != 1):
                dcorr = disco_loss(torch.from_numpy(classifications[key]), correlations, custom_weights)
                metrics[f"CLASSIFICATION/{key}_dcorr"] = dcorr

        ### Check whether all tracking metrics are available
        for item in self.options.tracking_metrics:
            if item not in metrics.keys():
                raise KeyError(
                    f"Metric {item} not available. Available metrics: {metrics.keys()}"
                )

        ### Log metrics
        for name, value in metrics.items():
            if not np.isnan(value):
                pbar_log = name in self.options.tracking_metrics
                if name == self.options.central_metric[0]: pbar_log = True
                self.log(name, value, sync_dist = True, on_epoch = True, prog_bar = pbar_log)

        self.logger.log_metrics(
            {name: value for name, value in metrics.items() if name in self.options.tracking_metrics},
            self.global_step
        )

        # self.validation_step_metrics_outputs.append(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

#    def on_validation_epoch_end(self):
#        # merge metrics from different mini batches into one dict
#        metrics_merged = defaultdict(list) 
#        for m in self.validation_step_metrics_outputs:
#            for key, value in m.items():
#                metrics_merged[key].append(value)
#
#        # average each metric over number of mini batches
#        metrics_averaged = {}
#        for key, values in metrics_merged.items():
#            metrics_averaged[f"mean_{key}"] = np.mean(values)
#
#        # log metrics
#        for name, value in metrics_averaged.items():
#            if not np.isnan(value):
#                self.log(name, value, sync_dist=True)
#
#        self.validation_step_metrics_outputs.clear()


