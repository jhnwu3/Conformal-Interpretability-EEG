import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
import csv
import pickle
from scipy import stats
from interpret.chefer import *
from abc import ABC, abstractmethod
import uq.kde 
import uq.covariate
from uq.class_conditional import *
from uq.conformity import *
from matplotlib.ticker import MaxNLocator
# not really a class object 
def print_prediction_set(pred_set, label_map=None):
    if label_map == None:
        print(pred_set)
    else: 
        to_print = []
        for i in range(len(pred_set)):
            to_print.append(label_map[int(pred_set[i].detach().cpu().numpy())])
        print(to_print)
    return to_print


# LORF implementation (?)

class ConformalCalibrator():
    def __init__(self, interpreter, method="softmax", kde_cal = None, kde_test = None, cal_dataloader=None):
        self.scorer = SoftMax(interpreter)
        self.method = method # for printing
        self.covariate = False
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
            self.covariate = True # for the sake of just keeping track of wtf is going on
            self.scorer = uq.covariate.SoftMaxCovariate(interpreter, kde_cal, kde_test, cal_dataloader)
        elif method == "cov_cum_softmax":
            self.covariate = True
            self.scorer = uq.covariate.CumulativeSoftMaxCovariate(interpreter, kde_cal, kde_test, cal_dataloader)
        else: 
            print("Error Need to Define a Scoring Function")
            exit(-1)

    # q_hat
    def calibrate(self, calibration_dataloader, alpha):
        # compute sorted calibration scores 
        conformity_scores = torch.tensor([]).to(self.scorer.interpreter.device)
        for signal, label in calibration_dataloader:
            # calibration scores should return a torch tensor
            score = self.scorer.cal_score(input=signal, class_index = label) # should return a scalar
            # print(score.size())
            conformity_scores = torch.cat((conformity_scores, score))
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache() # empty the cache before iterating
        
        # flatten and sort
        conformity_scores = conformity_scores.view(-1)
        # use the conformal prediction equation to calculate q_hat score
        # get quantile defined by alpha
        n = conformity_scores.size()[0]
        assert(int(1/alpha) < n) # throw error if alpha is too small

        q_level =  torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
        q_level = 1 - q_level # for conformity score not nonconformity score

        q_hat = torch.quantile(conformity_scores.to(q_level.dtype), q_level.to(self.scorer.interpreter.device), interpolation='lower')
        # detach and set to cpu to save VRAM
        return q_hat.detach().cpu()
 
    # get prediction set in torch
    def predict_set(self, input,  q_hat):
        scores = self.scorer.score(input)
        prediction_set = scores > q_hat
        return prediction_set.detach()
    
# do some post testing to see if anything from conformal prediciton can be interpreted
class InterpretableWrapper():
    def __init__(self, calibrator : ConformalCalibrator, interpretable_scorer):
        self.calibrator = calibrator
        self.val_scorer = interpretable_scorer
    


    # get only prediction sets > 1 
    def get_hard_set(self, prediction_sets, labels, inputs):

        greater_than_one_set = torch.nonzero(torch.sum(prediction_sets,dim=1) > 1)

        corresponding_labels = labels[greater_than_one_set.squeeze()]
        corresponding_inputs = inputs[greater_than_one_set.squeeze()]
        greater_than_one_set = prediction_sets[greater_than_one_set.squeeze()]
 
        return greater_than_one_set, corresponding_labels.view(-1), corresponding_inputs
    
    def get_easy_set(self, prediction_sets, labels, inputs):
        one_set = torch.nonzero(torch.sum(prediction_sets,dim=1) <= 1)
        corresponding_labels = labels[one_set.squeeze()]
        corresponding_inputs = inputs[one_set.squeeze()]
        one_set = prediction_sets[one_set.squeeze()]
        return one_set, corresponding_labels.view(-1), corresponding_inputs
    # input B x H x W (because this is EEG data)
    # only need to modify for B x C x H x W, probably later, all dependent on model design. 
    # actually want to 
    # returns B x 1 tensor of trues and falses, where true is if the interpretabiility score of the true label is the greatest
    def validate_hard_set(self, input, q_hat, label, method="confidence_drop"):
        # print(input.size())
        pred_set = self.calibrator.predict_set(input, q_hat)
        # get pred sets > 1
        large_pred_sets, hard_labels, hard_inputs = self.get_hard_set(pred_set, 
                                                           label.to(self.calibrator.scorer.interpreter.device),
                                                            input.to(self.calibrator.scorer.interpreter.device))
        # print("SHOULD BE THE SAME:", large_pred_sets.size())
        if (len(large_pred_sets.size()) < 2): # ugly bug checking
            large_pred_sets = large_pred_sets.unsqueeze(0)
        if (len(hard_labels.size()) < 1):
            hard_labels = hard_labels.unsqueeze(0)
        if (len(hard_inputs.size()) < 2):
            hard_inputs = hard_inputs.unsqueeze(0)

        assert(hard_inputs.size()[0] == large_pred_sets.size()[0] and hard_inputs.size()[0] == hard_labels.size()[0])

        validated = None
        if method == "confidence_drop":
            scores = self.val_scorer.score(hard_inputs)   # compute their interpretability scores
            
            # print("scores" , scores.size())
            validated = torch.argmax(scores, dim=1) == hard_labels
            # print("validated:", validated.size())
            # print("validated:", validated.size())
            # if largest, set ==true

            # get vectors of trues or falses


        # check if greatest score == label
        elif method == "similarity": # future plans
            pass
        # assert(validated.size()[0] == hard_inputs.size()[0])
        return validated.view(-1)
        # for i in range(input.size()[0]):
    def validate_hard_set_contains(self, input, q_hat, label, method="confidence_drop"):
        # assert(torch.sum(input - input.clone()) == 0)
       
       
        pred_set = self.calibrator.predict_set(input, q_hat)
        
        # get pred sets < 1
        easy_pred_sets, easy_labels, easy_inputs = self.get_hard_set(pred_set,
                                                                      label.to(pred_set.device), 
                                                                      input.to(pred_set.device))


        # assert(torch.sum(easy_pred_setss ^ easy_pred_sets) == 0)

        if (len(easy_pred_sets.size()) < 2): # ugly bug checking
            easy_pred_sets = easy_pred_sets.unsqueeze(0)
        
        if (len(easy_labels.size()) < 1):
            easy_labels = easy_labels.unsqueeze(0)
        
        if (len(easy_inputs.size()) < 2):
            easy_inputs = easy_inputs.unsqueeze(0)
        validated_num = 0
        validated_total = easy_pred_sets.size()[0]
        # print("hard_pred_sets:", easy_pred_sets.size())
        if method == "confidence_drop" and validated_total > 0:
           # go through each true in the prediction set,
            one_hots = F.one_hot(easy_labels, pred_set.size()[1])
            # print("ez:", easy_pred_sets.size())
            one_hots = easy_pred_sets * one_hots 
            validated_num = torch.sum(one_hots) 
            # print(validated_num.size())
        return validated_num, validated_total, pred_set
    
    def validate_easy_sets(self, input, q_hat, label, method="confidence_drop"):
        pred_set = self.calibrator.predict_set(input, q_hat)
        # get pred sets < 1
        easy_pred_sets, easy_labels, easy_inputs = self.get_easy_set(pred_set,
                                                                    label.to(pred_set.device), 
                                                                    input.to(pred_set.device))
        
        if (len(easy_pred_sets.size()) < 2): # ugly bug checking
            easy_pred_sets = easy_pred_sets.unsqueeze(0)
        
        if (len(easy_labels.size()) < 1):
            easy_labels = easy_labels.unsqueeze(0)
        
        if (len(easy_inputs.size()) < 2):
            easy_inputs = easy_inputs.unsqueeze(0)

        validated_num = 0
      
        validated_total = easy_pred_sets.size()[0]

        if method == "confidence_drop" and validated_total > 0:
           # go through each true in the prediction set,
            one_hots = F.one_hot(easy_labels, pred_set.size()[1])
         
            one_hots = easy_pred_sets * one_hots 
            validated_num = torch.sum(one_hots) 
     
        return validated_num, validated_total, pred_set

    
    

# in particular, we will just investigate softmax, because that makes the most sense currently.
def validate_interpretable(testloader, validator : InterpretableWrapper, q_hat):
    
    validated_hard = torch.empty(0)
    val_easy = 0
    val_easy_total = 0
    val_hard_in_sum = 0
    val_hard_total_sum = 0
    for batch, label in testloader:
        val_hard_in, val_hard_tot, pred_set = validator.validate_hard_set_contains(batch.clone(), q_hat, label)
     
        val_hard = validator.validate_hard_set(batch, q_hat, label)

        val_ez, val_ez_tot, pred_setss = validator.validate_easy_sets(batch.clone(), q_hat, label)

        assert(val_hard_tot == val_hard.size()[0])
        assert(val_hard_tot + val_ez_tot == batch.size()[0])
        
        
 
        validated_hard = torch.cat((validated_hard.to(val_hard.device), val_hard))
        val_easy += val_ez
        val_easy_total += val_ez_tot
        val_hard_in_sum += val_hard_in
        val_hard_total_sum += val_hard_tot
    validated_hard = validated_hard.view(-1)
    # sum to get total number of times ground truth is the most interpretable
    nTrue_Hard = torch.sum(validated_hard)
    acc_hard = nTrue_Hard / validated_hard.size()[0]
    acc_easy = val_easy / val_easy_total
    acc_hard_in = val_hard_in_sum / val_hard_total_sum
    print("dataset:", len(testloader.dataset))
    print("total validation easy:", val_easy_total)
    print("Total validation hard:", validated_hard.size()[0])
    print("total validation hard in:", val_hard_total_sum)
    return acc_hard, acc_easy, acc_hard_in


# # Get all metrics we need
# returns an accuracy sanity check, a mean set length, and a prediction set length 
def get_prediction_set_metrics(dataloader, calibrator, q_hat, n_classes=6):
    n_correct = 0
    prediction_set_lengths = torch.empty(0).to(calibrator.scorer.interpreter.device)
    # all_prediction_sets = torch.empty(0).to(calibrator.scorer.interpreter.device)
    # we should batch at some point, but for now this works. 
    mean_set_lengths = 0
    # also get class coverage
    class_coverage = torch.zeros(n_classes).to(calibrator.scorer.interpreter.device)
    total_per_class = torch.zeros(n_classes).to(calibrator.scorer.interpreter.device)
    for signal_batch, label in dataloader:
        # get one_hot labels 
        one_hot_labels = F.one_hot(label, n_classes).to(calibrator.scorer.interpreter.device).detach()
        prediction_set = calibrator.predict_set(signal_batch, q_hat) # N x C trues and falses for class per batch
        contains_ground_truth = prediction_set * one_hot_labels
        n_correct += torch.sum(contains_ground_truth).detach().cpu()
        # get class coverage
        class_coverage += torch.sum(contains_ground_truth, dim=0).detach().cpu()
        total_per_class += torch.sum(one_hot_labels, dim=0).detach().cpu()

        mean_set_lengths += torch.sum(prediction_set).detach().cpu().item() 
        # get prediction_set_lengths into a histogram
        prediction_set_lengths = torch.cat((prediction_set_lengths, torch.sum(prediction_set, dim=1).view(-1) ), axis=0) # stack flattened prediction_set_lengths
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache() # empty the cache before iterating
       
    mean_set_lengths = mean_set_lengths / float(len(dataloader.dataset))# divide total number of unique labels
    # all_prediction_sets = all_prediction_sets.view(-1) # flatten all of it into one giant array
    acc = float(n_correct) / len(dataloader.dataset)
    # detach to make sure we aren't causing any vram overflows!
    return acc, mean_set_lengths, prediction_set_lengths.view(-1).detach().cpu().numpy(), (class_coverage / total_per_class).cpu().numpy()

def convert_vec_to_dict(vector):
    # convert vector to dictionary
    dict = {}
    for i in range(len(vector)):
        dict[i] = vector[i]
    return dict


def write_to_csv(path, list1, list2, column_names):
     with open(path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)
        for values in zip(list1, list2):
            writer.writerow(values)
     csv_file.close()


def get_plots_research(test_loader, cal_loader, interpreter, n_classes=6, dataset='IIIC', normalize=False):
    logit_kde_cal = uq.kde.LogitDensityEstimator(interpreter, cal_loader, nEpochs=50)
    logit_kde_test = uq.kde.LogitDensityEstimator(interpreter, test_loader, nEpochs=50)
    rel_kde_cal = uq.kde.RelevanceDensityEstimator(interpreter, cal_loader, k=1, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_cal_IIIC.pt", use_labels=True)
    rel_kde_test = uq.kde.RelevanceDensityEstimator(interpreter, test_loader, k=1, class_index=None, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_test_IIIC.pt")
    
    calibrators = [
        ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=logit_kde_cal, kde_test=logit_kde_test, cal_dataloader=cal_loader),
        ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=rel_kde_cal, kde_test=rel_kde_test, cal_dataloader=cal_loader)
    ]
    q_hats = [f"uq/q_hat/IIIC/score_logit_cov_nbs{normalize}.csv",
              f"uq/q_hat/IIIC/score_rel_cov_nbs{normalize}.csv"]

    dfs = [] # read in all qhats
    for i in range (len(calibrators)):
        dfs.append(pd.read_csv(q_hats[i]))

    titles = ["Logit Covariate Score", "Relevance Covariate Score"]
    data_to_save = {}
    for i in range(len(calibrators)):
        cal = calibrators[i]
        df = dfs[i]
        alphas = df["alpha"]
        q_hats = df["q_hat"]
        print("Performing Prediction Sets for ", titles[i], " Calibrator")
        data_to_save[titles[i]] = { "alpha": np.array(alphas), "q_hat": np.array(q_hats)}
        # iterate through each q_hat and get back the metrics we care about
        accs = []
        avg_pset_lens = []
        pset_len = []
        class_coverages = []
        for q_hat in q_hats:
            acc, avg_pset_len, pset_lengths, class_coverage = get_prediction_set_metrics(test_loader, cal, q_hat, n_classes=n_classes)
            accs.append(acc)
            avg_pset_lens.append(avg_pset_len)
            pset_len.append(pset_lengths)
            class_coverages.append(class_coverage)
        
        data_to_save[titles[i]]["acc"] = np.array(accs)
        data_to_save[titles[i]]["avg_len"] = np.array(avg_pset_lens)
        data_to_save[titles[i]]["pset_len"] = np.array(pset_len)
        data_to_save[titles[i]]["cls_coverage"] = np.array(class_coverages)

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    with open(f'uq/metrics/IIIC/only_logit_rel_scores_cov_cls_plot_nbs{normalize_by_sample()}.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)
    
    # # create list of all calibrators we want
    # # NON-class specific runs!
    # print("Starting Non-Class Specific Prediction Sets")
    # rel_kde_cal = uq.kde.RelevanceDensityEstimator(interpreter, cal_loader, k=1, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_cal_IIIC.pt", use_labels=True)
    # rel_kde_test = uq.kde.RelevanceDensityEstimator(interpreter, test_loader, k=1, class_index=None, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_test_IIIC.pt")
    # attn_kde_cal = uq.kde.AttentionDensityEstimator(interpreter, cal_loader, nEpochs=50, saved_path="saved_weights/attn_kde_cal_IIIC.pt")
    # attn_kde_test = uq.kde.AttentionDensityEstimator(interpreter, test_loader, nEpochs=50, saved_path="saved_weights/attn_kde_test_IIIC.pt")
    # logit_kde_cal = uq.kde.LogitDensityEstimator(interpreter, cal_loader, nEpochs=50)
    # logit_kde_test = uq.kde.LogitDensityEstimator(interpreter, test_loader, nEpochs=50)
    # print("KDEs ESTIMATED")
    # calibrators = [ConformalCalibrator(interpreter, method="softmax"), 
    #                ConformalCovariateCalibrator(interpreter, method='softmax', kde_cal=attn_kde_cal, kde_test=attn_kde_test, cal_loader=cal_loader),
    #                ConformalCovariateCalibrator(interpreter, method='softmax', kde_cal=rel_kde_cal, kde_test=rel_kde_test, cal_loader=cal_loader),
    #                ConformalCovariateCalibrator(interpreter, method='softmax', kde_cal=logit_kde_cal, kde_test=logit_kde_test, cal_loader=cal_loader),
    #                ]
    # q_hats_non_class_specific = [f"uq/q_hat/IIIC/softmax_nbs{normalize}.csv", 
    #                              f"uq/q_hat/IIIC/attn_cov_nbs{normalize}.csv",
    #                              f"uq/q_hat/IIIC/rel_cov_nbs{normalize}.csv",
    #                              f"uq/q_hat/IIIC/logit_cov_nbs{normalize}.csv"]
    
    # titles_not_class_specific = ["Softmax", "Attention Covariate", "Relevance Covariate", "Logit Covariate"]
    # dfs = [] # read in all qhats
    # data_to_save = {}
    # acc_not_class_specific = []
    # avg_len_not_class_specific = []
    # pset_len_not_class_specific = []
    # class_coverage_not_class_specific = []

    # for i in range (len(calibrators)):
    #     dfs.append(pd.read_csv(q_hats_non_class_specific[i]))

    # for i in range(len(calibrators)):
    #     cal = calibrators[i]
    #     df = dfs[i]
    #     alphas = df["alpha"]
    #     q_hats = df["q_hat"]
    #     print("Performing Non-Class Specific Prediction Sets for ", titles_not_class_specific[i], " Calibrator")
    #     data_to_save[titles_not_class_specific[i]] = { "alpha": np.array(alphas), "q_hat": np.array(q_hats)}
    #     # iterate through each q_hat and get back the metrics we care about
    #     accs = []
    #     avg_pset_lens = []
    #     pset_len = []
    #     class_coverages = []
    #     for q_hat in q_hats:
    #         acc, avg_pset_len, pset_lengths, class_coverage = get_prediction_set_metrics(test_loader, cal, q_hat, n_classes=n_classes)
    #         accs.append(acc)
    #         avg_pset_lens.append(avg_pset_len)
    #         pset_len.append(pset_lengths)
    #         class_coverages.append(class_coverage)
        
    #     data_to_save[titles_not_class_specific[i]]["acc"] = np.array(accs)
    #     data_to_save[titles_not_class_specific[i]]["avg_len"] = np.array(avg_pset_lens)
    #     data_to_save[titles_not_class_specific[i]]["pset_len"] = np.array(pset_len)
    #     data_to_save[titles_not_class_specific[i]]["cls_coverage"] = np.array(class_coverages)

    #     # acc_not_class_specific.append(np.array(accs))
    #     # avg_len_not_class_specific.append(np.array(avg_pset_lens))
    #     # pset_len_not_class_specific.append(np.array(pset_len))
    #     # class_coverage_not_class_specific.append(np.array(class_coverages))
    #     gc.collect()
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()




    ######## CLASS_SPECIFIC STUFF!! ######
    # print("Starting Class Conditional Prediction Sets")
    # saved_paths_kde_test = {i : f"saved_weights/IIIC_aenc_kde_test_class{i}_nbs{normalize_by_sample()}.pt" for i in range(6)}
    # saved_paths_kde_cal = {i : f"saved_weights/IIIC_aenc_kde_cal_class{i}_nbs{normalize_by_sample()}.pt" for i in range(6)}
    
    # kde_cals, kde_tests = uq.kde.get_class_relevance_kdes(cal_loader, test_loader, interpreter,
    #                                                 saved_paths_cal= saved_paths_kde_cal,
    #                                                 saved_paths_test=saved_paths_kde_test)
    # # get all class specific calibrators


    # cc = ClassConformalCalibrator(interpreter, method="softmax")
    # cc_attn = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=attn_kde_cal, kde_test=attn_kde_test, cal_dataloader=cal_loader)
    # cc_rel = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=rel_kde_cal, kde_test=rel_kde_test, cal_dataloader=cal_loader)
    # # cc_logit = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=logit_kde_cal, kde_test=logit_kde_test, cal_dataloader=cal_loader)
    # # cc_covariate = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
    # cc_covariate_spcfc = ClassConformalCovariateCalibrator(interpreter, kde_cals, kde_tests, method="softmax", cal_loader=cal_loader)
    # cc_calibrators = [cc, cc_attn, cc_rel, cc_covariate_spcfc]

    # cls_base = f"uq/q_hat/IIIC/class_softmax_nbs{normalize_by_sample()}.csv"
    # cls_attn = f"uq/q_hat/IIIC/cls_attn_cov_nbs{normalize_by_sample()}.csv"
    # cls_rel = f"uq/q_hat/IIIC/cls_rel_cov_nbs{normalize_by_sample()}.csv"
    # # cls_logits = f"uq/q_hat/IIIC/cls_logit_cov_nbs{normalize_by_sample()}.csv"
    # # cls_cov = f"uq/q_hat/IIIC/class_cov_softmax_nbs{normalize_by_sample()}.csv"
    # cls_spcfc = f"uq/q_hat/IIIC/cls_rel_cov_spcfc_nbs{normalize_by_sample()}.csv"
    # cc_qhats = [cls_base, cls_attn, cls_rel, cls_spcfc]
    # cc_titles = ["Class Softmax",
    #               "Class Softmax with Attention Covariate",
    #                 "Class Softmax with Relevance Covariate",
    #                 #   "Class Softmax with Logit Covariate", 
    #                     "Class Conditional Softmax with Multiple Relevance KDEs Covariate Shift"]

    # alphas = [0.01, 0.05, 0.1, 0.15] 
    # cc_qhats_loaded = []
    # for qhat in cc_qhats:
    #     cc_qhats_loaded.append(np.loadtxt(qhat, delimiter=','))

    # # we will use a list of list of dictionaries
    # cc_qhat_dicts = []
    # for method in range(len(cc_qhats_loaded)):
    #     assert(cc_qhats_loaded[method].shape[1] == len(alphas))
    #     cc_qhat_dicts.append([])
    #     for a in range(len(alphas)):
    #         assert(cc_qhats_loaded[method].shape[0] == n_classes)
    #         cc_qhat_dicts[method].append(convert_vec_to_dict(cc_qhats_loaded[method][:,a]))
    # # get corresponding q_hat matrices for each alpha and class
    # # cc_base_q_hats = np.loadtxt(cls_base, delimiter=',') # need to implement this read and write process!
    # # cc_cov_q_hats = np.loadtxt(cls_cov, delimiter=',')
    # # cc_spcfc_q_hats = np.loadtxt(cls_spcfc, delimiter=',')

    # # cc_results = {}
    # for method in range(len(cc_qhat_dicts)):
    #     print("Performing Class Specific Prediction Sets for ", cc_titles[method], " Calibrator")
    #     data_to_save[cc_titles[method]] = {}
    #     assert(len(cc_qhat_dicts[method]) == len(alphas))
    #     cc_accs = []
    #     cc_mean_lens = []
    #     cc_lens = []
    #     cc_cls_coverages = []
    #     for a in range(len(alphas)):
    #         cc_acc, cc_mean_len, cc_len, cc_cls_coverage = get_prediction_set_metrics(test_loader, cc_calibrators[method], cc_qhat_dicts[method][a])
    #         cc_accs.append(cc_acc)
    #         cc_mean_lens.append(cc_mean_len)
    #         cc_lens.append(cc_len)
    #         cc_cls_coverages.append(cc_cls_coverage)

    #     data_to_save[cc_titles[method]] = {"acc": np.array(cc_accs), "avg_len": np.array(cc_mean_lens), "pset_len": np.array(cc_lens), "cls_coverage": cc_cls_coverages}

    # with open(f'uq/metrics/IIIC/fixed_cov_cls_plot_nbs{normalize_by_sample()}.pkl', 'wb') as file:
    #     pickle.dump(data_to_save, file)

    # cc_base_qhat_dicts = [] # should be across each alpha where the first elmt is all qhat corresponding to alpha 0.01
    # cc_cov_qhat_dicts = []
    # cc_spcfc_qhat_dicts = []
    # for i in range(cc_base_q_hats.shape[1]):
    #     cc_base_qhat_dicts.append(convert_vec_to_dict(cc_base_q_hats[:,i]))
    #     cc_cov_qhat_dicts.append(convert_vec_to_dict(cc_cov_q_hats[:,i]))
    #     cc_spcfc_qhat_dicts.append(convert_vec_to_dict(cc_spcfc_q_hats[:,i]))

    # get all metrics for each class
    # each column is an alpha 
    # each row is a class q_hat

    # cc_base_accs = [] # should be across each alpha where the first elmt is all qhat corresponding to alpha 0.01
    # cc_cov_accs = [] 
    # cc_spcfc_accs = []

    # cc_base_lens = []
    # cc_cov_lens = []
    # cc_spcfc_lens = []

    # cc_base_mean_lens = []
    # cc_cov_mean_lens = []
    # cc_spcfc_mean_lens = []
    
    # cc_base_cls_coverages = []
    # cc_cov_cls_coverages = []
    # cc_spcfc_cls_coverages = []

    # for i in range(len(cc_base_qhat_dicts)):
    #     cc_base_acc, cc_base_mean_len, cc_base_len, cc_base_cls_coverage = get_prediction_set_metrics(test_loader, cc, cc_base_qhat_dicts[i])
    #     cc_cov_acc, cc_cov_mean_len, cc_cov_len, cc_cov_cls_coverage = get_prediction_set_metrics(test_loader, cc_covariate, cc_cov_qhat_dicts[i])
    #     cc_spcfc_acc, cc_spcfc_mean_len, cc_spcfc_len, cc_spcfc_cov_cls_coverage = get_prediction_set_metrics(test_loader, cc_covariate_spcfc, cc_spcfc_qhat_dicts[i])
        
    #     cc_base_accs.append(cc_base_acc)
    #     cc_cov_accs.append(cc_cov_acc)
    #     cc_spcfc_accs.append(cc_spcfc_acc)

    #     cc_base_lens.append(cc_base_len)
    #     cc_cov_lens.append(cc_cov_len)
    #     cc_spcfc_lens.append(cc_spcfc_len)

    #     cc_base_cls_coverages.append(cc_base_cls_coverage)
    #     cc_cov_cls_coverages.append(cc_cov_cls_coverage)
    #     cc_spcfc_cls_coverages.append(cc_spcfc_cov_cls_coverage)

    #     cc_base_mean_lens.append(cc_base_mean_len)
    #     cc_cov_mean_lens.append(cc_cov_mean_len)
    #     cc_spcfc_mean_lens.append(cc_spcfc_mean_len)

        # compute the variances of the lengths

    # convert everything into a numpy array!
    # cc_base_accs = np.array(cc_base_accs)
    # cc_cov_accs = np.array(cc_cov_accs)
    # cc_spcfc_accs = np.array(cc_spcfc_accs)
    # cc_base_lens = np.array(cc_base_lens)
    # cc_cov_lens = np.array(cc_cov_lens)
    # cc_spcfc_lens = np.array(cc_spcfc_lens)
    # cc_base_mean_lens = np.array(cc_base_mean_lens)
    # cc_cov_mean_lens = np.array(cc_cov_mean_lens)
    # cc_spcfc_mean_lens = np.array(cc_spcfc_mean_lens)

    # cc_base_cls_coverages = np.array(cc_base_cls_coverages)
    # cc_cov_cls_coverages = np.array(cc_cov_cls_coverages)
    # cc_spcfc_cls_coverages = np.array(cc_spcfc_cls_coverages)

    # alphas = np.array(alphas)
    # avg_len_cov_softmax = np.array(avg_len_cov_softmax)
    # avg_len_softmax = np.array(avg_len_softmax)
    # acc_cov_softmax = np.array(acc_cov_softmax)
    # acc_softmax = np.array(acc_softmax)
    # pset_len_softmax = np.array(pset_len_softmax)
    # pset_len_cov_softmax = np.array(pset_len_cov_softmax)
    # class_coverage_softmax = np.array(class_coverage_softmax)
    # class_coverage_cov_softmax = np.array(class_coverage_cov_softmax)



    # # save all of the data in a giant pandas dataframe (for easy reconstruction)
    # df = {"alpha": np.array(alphas), 
    #                         "acc_softmax": acc_softmax, 
    #                         "acc_cov_softmax": acc_cov_softmax,
    #                           "avg_len_softmax": avg_len_softmax, 
    #                           "avg_len_cov_softmax": avg_len_cov_softmax, 
    #                           "cc_base_accs": cc_base_accs, 
    #                           "cc_cov_accs": cc_cov_accs,
    #                             "cc_spcfc_accs": cc_spcfc_accs, 
    #                             "cc_base_lens": cc_base_lens,
    #                               "cc_cov_lens": cc_cov_lens,
    #                               "cc_spcfc_lens": cc_spcfc_lens, 
    #                               "cc_base_mean_lens": cc_base_mean_lens, 
    #                               "cc_cov_mean_lens": cc_cov_mean_lens,
    #                                 "cc_spcfc_mean_lens": cc_spcfc_mean_lens,
    #                                 "pset_len_softmax": pset_len_softmax,
    #                                 "pset_len_cov_softmax": pset_len_cov_softmax,
    #                                 "class_coverage_softmax": class_coverage_softmax,
    #                                 "class_coverage_cov_softmax": class_coverage_cov_softmax,
    #                                 "cc_base_cls_coverages": cc_base_cls_coverages,
    #                                 "cc_cov_cls_coverages": cc_cov_cls_coverages,
    #                                 "cc_spcfc_cls_coverages": cc_spcfc_cls_coverages}
    
    # with open(f'uq/metrics/IIIC/cov_cls_plot_nbs{normalize_by_sample()}.pkl', 'wb') as file:
    #     pickle.dump(df, file)

    # calibrators = [ConformalCalibrator(interpreter, method="softmax"),
    #             #    ConformalCalibrator(interpreter, method="cum_softmax"),
    #             #    ConformalCalibrator(interpreter, method="softmax_drop"),
    #             #    ConformalCalibrator(interpreter, method="MORF"), 
    #             #    ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
    #                #ConformalCalibrator(interpreter, method="ROAD") # TAKES TOO LONG NEED TO IMPLEMENT OURSELVES
    #                ]
    
    # dfs = [] # read in all qhats
    # for cal in calibrators:
    #     dfs.append(pd.read_csv(f"uq/q_hat/{dataset}/{cal.method}_nbs{normalize}.csv"))
    # acc_plot = []
    # alpha_plot= []
    # lens_plot = []

    # acc_softmax = None
    # acc_cov_softmax = None
    # avg_len_softmax = None 
    # avg_len_cov_softmax = None
    # pset_len_softmax = None 
    # pset_len_cov_softmax = None
    # class_coverage_softmax = None 
    # class_coverage_cov_softmax = None
    # # should be in order
    # for i in range(len(calibrators)):
    #     cal = calibrators[i]
    #     df = dfs[i]
    #     alphas = df["alpha"]
    #     q_hats = df["q_hat"]
    #     # iterate through each q_hat and get back the metrics we care about
    #     accs = []
    #     avg_pset_lens = []
    #     pset_len = []
    #     class_coverages = []
    #     i = 0
    #     for q_hat in q_hats:
    #         acc, avg_pset_len, pset_lengths, class_coverage = get_prediction_set_metrics(test_loader, cal, q_hat, n_classes=n_classes)
    #         accs.append(acc)
    #         avg_pset_lens.append(avg_pset_len)
    #         pset_len.append(pset_lengths)
    #         class_coverages.append(class_coverage)
    #         # plt.figure() 
    #         # plt.title(cal.method + " alpha" + str(alphas[i])) 
    #         # plt.hist(pset_lengths,bins=n_classes)
    #         # plt.ylabel("# of Samples")
    #         # plt.xlabel("Prediction Set Lengths")
    #         # plt.savefig(f'fig/cp/{dataset}/{cal.method}/hist__alpha' + str(alphas[i]) + "_nbs"+ str(normalize)+ ".png")
    #         i+=1

    #     # get the ones we want ot compare to!
    #     if cal.covariate == False:
    #         acc_softmax = accs 
    #         avg_len_softmax = avg_pset_lens
    #         pset_len_softmax = pset_len
    #         class_coverage_softmax = class_coverages
    #     elif cal.covariate == True:
    #         acc_cov_softmax = accs
    #         avg_len_cov_softmax = avg_pset_lens
    #         pset_len_cov_softmax = pset_len
    #         class_coverage_cov_softmax = class_coverages

    #     alpha_plot.append(np.array(alphas))
    #     acc_plot.append(np.array(accs))
    #     lens_plot.append(np.array(avg_pset_lens))
        
    #     gc.collect()
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()

    # UNCOMMENT TO DO REPRODUCE THINGS NORMALLY!
    ### plot alpha vs. acc  plots for all calibration scores ###
    # plt.title("Accuracies") # this should all be constant
    # plt.figure()
    # for i in range(len(calibrators)):
    #     plt.ylabel("Coverage")
    #     plt.xlabel("Alpha")
    #     print(calibrators[i].method)
    #     print("acc:", acc_plot[i])
    #     tosave = np.concatenate((alpha_plot[i].reshape(-1,1), acc_plot[i].reshape(-1,1)), axis=1)
    #     np.savetxt(f"uq/acc/{calibrators[i].method}",tosave.numpy(), delimiter=",") # save to csv

    #     plt.plot(alpha_plot[i], acc_plot[i], label=calibrators[i].method)

        
    
    # plt.plot(alpha_plot[0], 1 - alpha_plot[0], label="Ideal")
    # plt.legend()
    # plt.savefig(f"fig/cp/{dataset}/acc_all_nbs{normalize}.png")
   
    # # plot alpha vs average prediciton set lengths for all calibration scores
    # plt.figure()
    # plt.title("Prediction Set Lengths")
    # for i in range(len(calibrators)):
    #     # alphas = alpha_plot[i]
    #     plt.ylabel("Average Prediction Set Length")
    #     plt.xlabel("Alpha")
    #     plt.plot(alpha_plot[i], lens_plot[i], label=calibrators[i].method)
    #     tosave = np.concatenate((alpha_plot[i].reshape(-1,1), lens_plot[i].reshape(-1,1)), axis=1)
    #     np.savetxt(f"uq/acc/{calibrators[i].method}",tosave , delimiter=",") # save to csv
    # plt.legend()
    # plt.savefig(f"fig/cp/{dataset}/avg_set_len_all_nbs{normalize}.png")



    # plot alpha vs. acc  plots for all important baselines 
    # plt.figure(figsize=(10,10)) # big plot
    # plt.plot(alphas, cc_base_accs, label="Class Softmax")
    # plt.plot(alphas, cc_cov_accs, label="Class Softmax with General Covariate Shift")
    # plt.plot(alphas, cc_spcfc_accs, label="Class Conditional Softmax with Specific Covariate Shift")
    # plt.plot(alphas, acc_softmax, label="Base Softmax")
    # plt.plot(alphas, acc_cov_softmax, label="Softmax with General Covariate Shift")
    # plt.plot(alphas, [1 - alpha for alpha in alphas], label="Ideal")
    # plt.xlabel("Alpha")
    # plt.ylabel("Coverage")
    # plt.legend()
    # plt.savefig("fig/cp/IIIC/acc_all_cls_nbs" + str(normalize_by_sample()) + ".png")
    # # plot alpha vs. acc  plots for all important baselines 
    # plt.figure(figsize=(10,10)) # big plot
    # plt.xlabel("Alpha")
    # plt.ylabel("Average Prediction Set Length")
    # plt.plot(alphas, cc_base_mean_lens, label="Class Softmax")
    # plt.plot(alphas, cc_cov_mean_lens, label="Class Softmax with General Covariate Shift")
    # plt.plot(alphas, cc_spcfc_mean_lens, label="Class Conditional Softmax with Specific Covariate Shift")
    # plt.plot(alphas, avg_len_softmax, label="Base Softmax")
    # plt.plot(alphas, avg_len_cov_softmax, label="Softmax with General Covariate Shift")
    # plt.legend()
    # plt.savefig("fig/cp/IIIC/mean_len_cls_nbs" + str(normalize_by_sample()) + ".png")

    # # Do a ratio between the two mean length vs. acc plots, acc / mean length 
    # plt.figure(figsize=(10,10)) # big plot
    # plt.xlabel("Alpha")
    # plt.ylabel("Accuracy / Average Prediction Set Length")
    # plt.plot(alphas, cc_base_accs / cc_base_mean_lens, label="Class Softmax")
    # plt.plot(alphas, cc_cov_accs / cc_cov_mean_lens, label="Class Softmax with General Covariate Shift")
    # plt.plot(alphas, cc_spcfc_accs / cc_spcfc_mean_lens, label="Class Conditional Softmax with Specific Covariate Shift")
    # plt.plot(alphas, acc_softmax / avg_len_softmax, label="Base Softmax")
    # plt.plot(alphas, acc_cov_softmax / avg_len_cov_softmax, label="Softmax with General Covariate Shift")
    # plt.legend()
    # plt.savefig("fig/cp/IIIC/ratio_acc_len_cls_nbs" + str(normalize_by_sample()) + ".png")

    # print("cc_base Shape:", cc_base_lens.shape)

    # # Compute their variances to see how much they vary across the different alphas (More variance usually better)
    # plt.figure(figsize=(10,10)) # big plot
    # plt.xlabel("Alpha")
    # plt.ylabel("Variance of Prediction Set Lengths")
    # plt.plot(alphas, np.var(cc_base_lens, axis=1), label="Class Softmax")
    # plt.plot(alphas, np.var(cc_cov_lens, axis=1), label="Class Softmax with General Covariate Shift")
    # plt.plot(alphas, np.var(cc_spcfc_lens, axis=1), label="Class Conditional Softmax with Specific Covariate Shift")
    # plt.plot(alphas, np.var(pset_len_softmax, axis=1), label="Base Softmax")
    # plt.plot(alphas, np.var(pset_len_cov_softmax, axis=1), label="Softmax with General Covariate Shift")
    # plt.legend()
    # plt.savefig("fig/cp/IIIC/var_cls_nbs" + str(normalize_by_sample()) + ".png")

# both should be np arrays
def compute_KL_distance(prediction_set1, prediction_set2):
    # compute counts of each to get some metric of which classes it's more likely to predict
    count_labels1 = np.bincount(prediction_set1)
    count_labels2 = np.bincount(prediction_set2)
    count_labels1 = count_labels1 / np.sum(count_labels1)
    count_labels2 = count_labels2 / np.sum(count_labels2)

    # compute KL distance
    kl_div = np.sum(count_labels1 * np.log(count_labels1 / count_labels2))
    return kl_div
