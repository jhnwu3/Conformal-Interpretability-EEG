import scipy.stats 
import numpy as np
import torch
import bisect
from data import *
from models.autoencoder import *
from interpret.chefer import *
from einops import rearrange
import uq.kde
from typing import Dict, Union
from torch.utils.data import Subset
from abc import ABC, abstractmethod
from uq.conformity import *
import uq.covariate 
# from uq.covariate import *
# class ClassCovariateSoftMax(SoftMaxCovariate):
#     def __init__(self, interpreter, kde_cals, kde_tests, cal_dataloader):
#         self.interpreter = interpreter
#         self.kde_cals  = kde_cals # should be a dict of class indices and density estimators
#         self.kde_tests = kde_tests 
#         self.cal_dataloader = cal_dataloader # for summing up the likelihood ratios
#         self.sum_cal_likelihoods = {key: sum_likelihood_ratios(self.cal_dataloader, value, self.kde_tests[key]) for key, value in self.kde_cals.items()}
#     def cal_score(input, class_index):
#         # basically we want to weigh the softmax score by the likelihood ratio


#     def score(self, input):



# class balanced calibration
class ClassConformalCalibrator():
    def __init__(self, interpreter, method="softmax", kde_cal = None, kde_test = None, cal_dataloader=None):
        self.scorer = None
        self.method = method # for printing
        if method == "softmax":
            self.scorer= SoftMax(interpreter)
        elif method == "cum_softmax":
            self.scorer = CumulativeSoftMax(interpreter)
        elif method == "softmax_drop":
            self.scorer = SoftMaxDrop(interpreter)
        elif method == "ROAD":
            self.scorer = ROAD(interpreter)
        elif method =="MORF":
            self.scorer = MORF(interpreter)
        elif method == "cov_softmax":
            self.scorer = uq.covariate.SoftMaxCovariate(interpreter, kde_cal, kde_test, cal_dataloader)
        elif method == "cov_cum_softmax":
            self.scorer = uq.covariate.CumulativeSoftMaxCovariate(interpreter, kde_cal, kde_test, cal_dataloader)
        elif method == "cov_softmax_quantile":
            self.scorer = SoftMax(interpreter)
            self.kde_cal = kde_cal 
            self.kde_test = kde_test
        else: 
            print("Error Need to Define a Scoring Function")
            exit(-1)

    # q_hat
    def calibrate(self, calibration_dataloader, alpha, n_classes=6):
        # compute sorted calibration scores 
        conformity_scores = torch.tensor([]).to(self.scorer.interpreter.device)
        calibration_weights = torch.tensor([]).to(self.scorer.interpreter.device)
        sum_likelihoods = 0
        if self.method == "cov_softmax_quantile":
            sum_likelihoods = uq.covariate.sum_likelihood_ratios(calibration_dataloader, self.kde_cal, self.kde_test)

        # calibrate for each class 
        all_labels = []
        for signal, label in calibration_dataloader:
            # calibration scores should return a torch tensor
            score = self.scorer.cal_score(input=signal, class_index = label) # should return a scalar
            if self.method == "cov_softmax_quantile": # bad code hahaha
                calibration_weight = uq.covariate.calibration_weight(signal, sum_likelihoods, self.kde_cal, self.kde_test)
                calibration_weights = torch.cat((calibration_weights, calibration_weight))

            # print(score.size())
            conformity_scores = torch.cat((conformity_scores, score))
            all_labels.append(label)
            gc.collect()
        
            with torch.no_grad():
                torch.cuda.empty_cache() # empty the cache before iterating
        
        # flatten and sort
        all_labels = torch.cat(all_labels).view(-1)
       
        conformity_scores = conformity_scores.view(-1)
        # break into 6 pools of conformity scores
        unique_labels = torch.unique(all_labels)
        class_cal_scores = {label.item(): conformity_scores[all_labels == label] for label in unique_labels}
        class_weights = {}
        if self.method == "cov_softmax_quantile":
            class_weights = {label.item(): calibration_weights[all_labels == label] for label in unique_labels}
        q_hats = {} # get dictionary of class : q_hat pairs
        for key, value in class_cal_scores.items():
            n = value.size()[0]
            assert(int(1/alpha) < n) # throw error if alpha is too small
        
            q_level = torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
            q_level = 1 - q_level # for conformity score not nonconformity score
            if self.method == "cov_softmax_quantile":
                q_hat = _query_weighted_quantile_torch(value, q_level, class_weights[key])
            else:
                q_hat = torch.quantile(value.to(q_level.dtype), q_level.to(self.scorer.interpreter.device), interpolation='lower')
            q_hats[key] = q_hat.detach().cpu()
       
        return q_hats
 
    # get prediction set in torch
    def predict_set(self, input, q_hats):
     
        scores = self.scorer.score(input)
        # go through each score column (i.e class)
        if len(scores.size()) == 1: # i.e only 1 example
            scores = scores.unsqueeze(0)
        for key, q_hat in q_hats.items():
            # print("what's happening:", scores.size())
            scores[:, key] = scores[:, key] > q_hat
        # prediction_set = scores > q_hats
        return scores.detach()






class ConformalCovariateCalibrator():
    def __init__(self, interpreter, kde_cal, kde_test, cal_loader, method="softmax"):
        self.scorer = None
        self.kde_cal = kde_cal
        self.kde_test = kde_test
        self.method = method # for printing
        self.covariate = True

        # basically we want to keep a dictionary of class specific likelihood ratios and datasets
        self.class_batches = None 
        self.sum_likelihoods = None
        self.all_labels = []
        self.all_signals = []
        for signal, label in cal_loader:
            self.all_labels.append(label)
            self.all_signals.append(signal)

        self.all_labels = torch.cat(self.all_labels).view(-1)
        self.all_signals = torch.cat(self.all_signals).view(-1)
        # self.unique_labels = torch.unique(self.all_labels)
        # self.class_batches = {label.item(): all_signals[self.all_labels == label] for label in self.unique_labels}

        if method == "softmax":
            self.scorer= SoftMax(interpreter)
        elif method == "cum_softmax":
            self.scorer = CumulativeSoftMax(interpreter)
        elif method == "softmax_drop":
            self.scorer = SoftMaxDrop(interpreter)
        elif method == "ROAD":
            self.scorer = ROAD(interpreter)
        elif method =="MORF":
            self.scorer = MORF(interpreter)
        else: 
            print("Error Need to Define a Scoring Function")
            exit(-1)

    # q_hat
    def calibrate(self, calibration_dataloader, alpha, n_classes=6):
        # compute sorted calibration scores 
        conformity_scores = torch.tensor([]).to(self.scorer.interpreter.device)
        calibration_weights = torch.tensor([]).to(self.scorer.interpreter.device)
        sum_likelihoods = uq.covariate.sum_likelihood_ratios(calibration_dataloader, self.kde_cal, self.kde_test) 
        # calibrate for each class 
        for signal, label in calibration_dataloader:
            # calibration scores should return a torch tensor
            score = self.scorer.cal_score(input=signal, class_index = label) # should return a list of scalars
            calibration_weight = uq.covariate.calibration_weight(signal, sum_likelihoods, self.kde_cal, self.kde_test)
            # print(score.size())
            conformity_scores = torch.cat((conformity_scores, score))
            calibration_weights = torch.cat((calibration_weights, calibration_weight))
            # print(label.size())
            gc.collect()
        
            with torch.no_grad():
                torch.cuda.empty_cache() # empty the cache before iterating
        
        # flatten and sort
        # all_labels = torch.cat(all_labels).view(-1)
        # all_signals = torch.cat(all_signals)

        calibration_weights = calibration_weights.view(-1)
        conformity_scores = conformity_scores.view(-1)

        # break into 6 pools of conformity scores
        # unique_labels = torch.unique(all_labels)

        # class_cal_scores = {label.item(): conformity_scores[self.all_labels == label] for label in self.unique_labels}
        # weigh each of the scores by their respective likelihood ratios
        n = conformity_scores.size()[0]
        q_level =  torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
        q_level = 1 - q_level # for conformity score not nonconformity score
        # class_cal_weights = {}
        q_hat = _query_weighted_quantile_torch(conformity_scores.squeeze(), q_level, calibration_weights.squeeze())
        # for key, scores in class_cal_scores.items():
        #     n = scores.size()[0]
        #     assert(int(1/alpha) < n) # throw error if alpha is too small
        #     # q_level =  torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
        #     # q_level = 1 - q_level # for conformity score not nonconformity score
        #     # q_hat = torch.quantile(value.to(q_level.dtype), q_level.to(self.scorer.interpreter.device), interpolation='lower')
            
        #     q_hats[key] = q_hat.detach().cpu()
        return q_hat
 
    # get prediction set in torch
    def predict_set(self, input, q_hat):
        scores = self.scorer.score(input)
        if len(scores.size()) == 1: # i.e only 1 example
            scores = scores.unsqueeze(0)
        # go through each score column (i.e class)
        prediction_set = scores > q_hat
        return prediction_set.detach()

# note that the density estimators we want are explicitly of the form p(x|y) where y is the class label
# and that they are using chefer's relevance rather than attention scores
class ClassConformalCovariateCalibrator():
    def __init__(self, interpreter, kde_cals, kde_tests, cal_loader, method="softmax"):
        self.scorer = None
        self.kde_cals = kde_cals 
        self.kde_tests = kde_tests
        self.method = method # for printing
        
        # basically we want to keep a dictionary of class specific likelihood ratios and datasets
        self.class_batches = None 
        self.sum_likelihoods = None
        self.all_labels = []
        all_signals = []
        for signal, label in cal_loader:
            self.all_labels.append(label)
            all_signals.append(signal)

        self.all_labels = torch.cat(self.all_labels).view(-1)
        all_signals = torch.cat(all_signals)
        self.unique_labels = torch.unique(self.all_labels)
        self.class_batches = {label.item(): all_signals[self.all_labels == label] for label in self.unique_labels}
        self.sum_likelihoods = {key: uq.covariate.sum_likelihood_ratios( 
                                    torch.utils.data.DataLoader(uq.kde.ClassSpecificDataset(self.class_batches[key], torch.tensor(key)), batch_size=1000, shuffle=True, num_workers=32), kde, self.kde_tests[key]) 
                                for key, kde in self.kde_cals.items()}


        if method == "softmax":
            self.scorer= SoftMax(interpreter)
        elif method == "cum_softmax":
            self.scorer = CumulativeSoftMax(interpreter)
        elif method == "softmax_drop":
            self.scorer = SoftMaxDrop(interpreter)
        elif method == "ROAD":
            self.scorer = ROAD(interpreter)
        elif method =="MORF":
            self.scorer = MORF(interpreter)
        else: 
            print("Error Need to Define a Scoring Function")
            exit(-1)

    # q_hat
    def calibrate(self, calibration_dataloader, alpha, n_classes=6):
        # compute sorted calibration scores 
        conformity_scores = torch.tensor([]).to(self.scorer.interpreter.device)

        # calibrate for each class 
        for signal, label in calibration_dataloader:
            # calibration scores should return a torch tensor
            score = self.scorer.cal_score(input=signal, class_index = label) # should return a scalar
            
            # print(score.size())
            conformity_scores = torch.cat((conformity_scores, score))
            # print(label.size())
            gc.collect()
        
            with torch.no_grad():
                torch.cuda.empty_cache() # empty the cache before iterating
        
        # flatten and sort
        # all_labels = torch.cat(all_labels).view(-1)
        # all_signals = torch.cat(all_signals)

        
        conformity_scores = conformity_scores.view(-1)

        # break into 6 pools of conformity scores
        # unique_labels = torch.unique(all_labels)

        class_cal_scores = {label.item(): conformity_scores[self.all_labels == label] for label in self.unique_labels}
        # weigh each of the scores by their respective likelihood ratios

        
        class_cal_weights = {}
        for key, value in class_cal_scores.items():
            n = value.size()[0]
            assert(int(1/alpha) < n) # throw error if alpha is too small
            # Now let's weigh each score in the class cal scores by the likelihood ratio
            class_cal_weights[key] = uq.covariate.calibration_weight(self.class_batches[key], self.sum_likelihoods[key], self.kde_cals[key], self.kde_tests[key])

        q_hats = {} # get dictionary of class : q_hat pairs
        
        for key, scores in class_cal_scores.items():
            n = scores.size()[0]
            assert(int(1/alpha) < n) # throw error if alpha is too small
            q_level =  torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
            q_level = 1 - q_level # for conformity score not nonconformity score
            # q_hat = torch.quantile(value.to(q_level.dtype), q_level.to(self.scorer.interpreter.device), interpolation='lower')
            q_hat = _query_weighted_quantile_torch(scores=scores.squeeze(), alpha=q_level, weights=class_cal_weights[key].squeeze())
            q_hats[key] = q_hat.detach().cpu()

        return q_hats
 
    # get prediction set in torch
    def predict_set(self, input, q_hats):
     
        scores = self.scorer.score(input)
        if len(scores.size()) == 1: # i.e only 1 example
            scores = scores.unsqueeze(0)
        # go through each score column (i.e class)
        for key, q_hat in q_hats.items():
            # the test weights should be class specific!
            # scores[:, key] = (uq.covariate.test_weight(input, self.sum_likelihoods[key], self.kde_cals[key], self.kde_tests[key]) * scores[:, key]) > q_hat
            scores[:, key] = (scores[:, key]) > q_hat
        # prediction_set = scores > q_hats
        return scores.detach()
    




# helper functions
def _query_quantile(scores, alpha):
    scores = np.sort(scores)
    N = len(scores)
    loc = int(np.floor(alpha * (N + 1))) - 1
    return -np.inf if loc == -1 else scores[loc]

def _query_weighted_quantile_torch(scores, alpha, weights):
    # in our case since we've flipped it, alpha will find that alpha smallest conformal score 
    # should be fine because of exchangeability
    sorted_scores, sorted_idxs = torch.sort(scores) # sort before search
    cum_weights = torch.cumsum(weights[sorted_idxs], dim=0) / torch.sum(weights)
    idx = torch.searchsorted(cum_weights, alpha) + 1
    if idx > len(sorted_scores): # bound it to be at least 0, although for now it seems fine. 
        idx = len(sorted_scores) - 1
    return sorted_scores[idx]

def _query_weighted_quantile(scores, alpha, weights):
    # should be sorted in the order of the scores
    qs = np.cumsum(weights)/np.sum(weights) 
    idx = bisect.bisect_left(qs, alpha, lo=0, hi=len(qs)-1)
    return scores[idx]

# class WeightedCP(SetPredictor):
#     """LABEL: Least ambiguous set-valued classifiers with bounded error levels.
#     This is a prediction-set constructor for multi-class classification problems.
#     It controls either :math:`\\mathbb{P}\\{Y \\not \\in C(X) | Y=k\\}\\leq \\alpha_k`
#     (when ``alpha`` is an array), or :math:`\\mathbb{P}\\{Y \\not \\in C(X)\\}\\leq \\alpha` (when ``alpha`` is a float).
#     Here, :math:`C(X)` denotes the final prediction set.
#     This is essentially a split conformal prediction method using the predicted scores.
#     Paper:
#         Sadinle, Mauricio, Jing Lei, and Larry Wasserman.
#         "Least ambiguous set-valued classifiers with bounded error levels."
#         Journal of the American Statistical Association 114, no. 525 (2019): 223-234.
#     :param model: A trained base model.
#     :type model: BaseModel
#     :param alpha: Target mis-coverage rate(s).
#     :type alpha: Union[float, np.ndarray]
#     Examples:
#         >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
#         >>> from pyhealth.models import SparcNet
#         >>> from pyhealth.tasks import sleep_staging_isruc_fn
#         >>> from pyhealth.calib.predictionset import LABEL
#         >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
#         >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
#         >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
#         ...     label_key="label", mode="multiclass")
#         >>> # ... Train the model here ...
#         >>> # Calibrate the set classifier, with different class-specific mis-coverage rates
#         >>> cal_model = LABEL(model, [0.15, 0.3, 0.15, 0.15, 0.15])
#         >>> # Note that we used the test set here because ISRUCDataset has relatively few
#         >>> # patients, and calibration set should be different from the validation set
#         >>> # if the latter is used to pick checkpoint. In general, the calibration set
#         >>> # should be something exchangeable with the test set. Please refer to the paper.
#         >>> cal_model.calibrate(cal_dataset=test_data)
#         >>> # Evaluate
#         >>> from pyhealth.trainer import Trainer, get_metrics_fn
#         >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
#         >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(test_dl, additional_outputs=['y_predset'])
#         >>> print(get_metrics_fn(cal_model.mode)(
#         ... y_true_all, y_prob_all, metrics=['accuracy', 'miscoverage_ps'],
#         ... y_predset=extra_output['y_predset'])
#         ... )
#         {'accuracy': 0.709843241966832, 'miscoverage_ps': array([0.1499847 , 0.29997638, 0.14993964, 0.14994704, 0.14999252])}
#     """

#     def __init__(
#         self, 
#         model, 
#         alpha: Union[float, np.ndarray], 
#         debug=False,
#          **kwargs
#     ) -> None:
#         super().__init__(model, **kwargs)
#         if model.mode != "multiclass":
#             raise NotImplementedError()
#         self.mode = self.model.mode  # multiclass
#         for param in model.parameters():
#             param.requires_grad = False
#         self.model.eval()
#         self.device = model.device
#         self.debug = debug
#         if not isinstance(alpha, float):
#             alpha = np.asarray(alpha)
#         self.alpha = alpha

#         self.t = None

#     def calibrate(self, cal_dataset: Union[Subset, Dict], weights=None, calibration_type="inductive"):
#         """Calibrate the thresholds used to construct the prediction set.
#         :param cal_dataset: Calibration set.
#         :type cal_dataset: Subset
#         """
#         assert (calibration_type == "mondrian" or calibration_type == "inductive", "Expection calibration type is inductive or mondrian. Got: {calibration_type}")

#         if not isinstance(cal_dataset, dict):
#             cal_dataset = prepare_numpy_dataset(
#                 self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
#             )

#         y_prob = cal_dataset["y_prob"]
#         y_true = cal_dataset["y_true"]

#         if weights is None:
#             weights = [1.0 for i in range(len(y_prob))]
#             logging.warning("No weights specified for calibration. Calibrating with equal weights for all samples")

#         assert (len(weights) == len(y_prob), f"Weights should be of same size as y_prob, Expected {len(y_prob)} weights has {len(weights)} values")

#         N, K = cal_dataset["y_prob"].shape
#         if calibration_type == "inductive":
#             if isinstance(self.alpha, float):
#                 t = _query_weighted_quantile(y_prob[np.arange(N), y_true], self.alpha, weights)
#             else:
#                 t = [
#                     _query_weighted_quantile(y_prob[np.arange(N), y_true], self.alpha[k], weights) for k in range(K)
#                 ]

#         self.t = torch.tensor(t, device=self.device)


#     def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
#         """Forward propagation (just like the original model).
#         :return: A dictionary with all results from the base model, with the following updates:
#                     y_predset: a bool tensor representing the prediction for each class.
#         :rtype: Dict[str, torch.Tensor]
#         """
#         pred = self.model(**kwargs)
#         pred["y_predset"] = pred["y_prob"].cpu() > self.t
#         return pred