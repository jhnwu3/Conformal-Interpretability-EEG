import scipy.stats 
import numpy as np
import torch
import csv
import time
from data import *
from models.autoencoder import *
from interpret.chefer import *
from einops import rearrange
from uq.kde import *
from uq.covariate import *
from uq.conformal import *
from uq.class_conditional import *


def qhat_dict_to_vector(q_hat_dict):
    # get max class index + 1
    print(q_hat_dict.keys())
    len_vec = np.max(np.array(list(q_hat_dict.keys()))) + 1
    vec = np.zeros(len_vec)
    for key, values in q_hat_dict.items():
        vec[key] = values
    return vec 

def write_to_csv(path, list1, list2, column_names):
     with open(path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)
        for values in zip(list1, list2):
            writer.writerow(values)
     csv_file.close()

class_labels = {
    0 : 'SPSW', 1 : 'GPED', 2 : 'PLED', 3 : 'EYEM', 4: 'ARTF', 5 :'BCKG'
}

# my model hyperparameters
emb_size = 256
depth = 6 
dropout = 0.5
num_heads = 8
patch_kernel_length = 11  # cqi = 15 - UNUSED
stride = 11  # cqi = 8 - UNUSE
model = STTransformer(emb_size=emb_size, 
                                depth=depth,
                                n_classes=6, 
                                channel_length=2000,
                                dropout=dropout, 
                                num_heads=num_heads,
                                kernel_size=11, 
                                stride=11,
                                kernel_size2=11,
                                stride2=11)

model.load_state_dict(torch.load(f"saved_weights/st_transformer_conformal_IIIC_nbs{normalize_by_sample()}.pt"))


# my training hyperparameters
sampling_rate = 200
batch_size = 500 # may be too memory intensive depending on the conformity score...
num_workers = 8
train_loader, test_loader, val_loader, cal_loader = prepare_IIIC_cal_dataloader(batch_size=batch_size, num_workers=num_workers, sample_norm=normalize_by_sample())

# get interpreter
interpreter = STTransformerInterpreter(model=model)


alphas = [0.01, 0.05, 0.1, 0.15] 
column_names = ["alpha", "q_hat"]
signal, label = test_loader.dataset[0]
signal = signal.unsqueeze(0)
current_gpu = torch.cuda.current_device()
print("Currently active GPU:", current_gpu)
q_hats = []
# print("GradCam++ Softmax Drop")
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache() # empty the cache before iterating

# cal = ConformalCalibrator(interpreter,method="softmax_drop")
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))


# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/softmax_drop_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)
# gc.collect()
# with torch.no_grad():
#     torch.cuda.empty_cache() # empty the cache before iterating

# q_hats = []
# we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
# print("SOFTMAX")
# cal = ConformalCalibrator(interpreter, method="softmax")
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))
#     # all_pred_sets, acc, mean_pred_length , set_lengths = get_prediction_set_metrics(cal_loader, cal, q_hat)
#     # print("acc:", acc)
#     # print("mean pred set length:", mean_pred_length)

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/softmax_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)
# gc.collect()
# with torch.no_grad():
#     torch.cuda.empty_cache() # empty the cache before iterating

# q_hats = []
# # we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
# print("MORF")
# cal = ConformalCalibrator(interpreter, method="MORF")
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/MORF_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)


# q_hats = []
# # we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
# print("cumsum SOFTMAX")
# cal = ConformalCalibrator(interpreter, method="cum_softmax")
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/cum_softmax_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)

# start from the top and debug!

# print("Attention Covariate Shift Test Run!")
# q_hats = []
# kde_cal = AttentionDensityEstimator(interpreter, cal_loader, nEpochs=50, saved_path="saved_weights/attn_kde_cal_IIIC.pt")
# kde_test = AttentionDensityEstimator(interpreter, test_loader, nEpochs=50, saved_path="saved_weights/attn_kde_test_IIIC.pt")
# print("Attention Density Estimated!")
# cal = ConformalCovariateCalibrator(interpreter, method="softmax", kde_cal=kde_cal, kde_test=kde_test, cal_loader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/test_attn_cov_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)


# print("Class Conditional Attention Covariate Shift Test Run!")
# q_hats = []
# kde_cal = AttentionDensityEstimator(interpreter, cal_loader, nEpochs=50, saved_path="saved_weights/attn_kde_cal_IIIC.pt")
# kde_test = AttentionDensityEstimator(interpreter, test_loader, nEpochs=50, saved_path="saved_weights/attn_kde_test_IIIC.pt")
# cal = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# q_hats = np.array(q_hats).transpose()
# print("Time:", time.time() - start)
# np.savetxt(f"uq/q_hat/IIIC/cls_attn_cov_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")


# print("Relevance Covariate Shift Test Run!")
# q_hats = []
# kde_cal = RelevanceDensityEstimator(interpreter, cal_loader, k=1, nEpochs=50, kde_dim=6, to_save_path="saved_weights/rel_kde_cal_IIIC.pt", use_labels=True)
# kde_test = RelevanceDensityEstimator(interpreter, test_loader, k=1, class_index=None, nEpochs=50, kde_dim=6, to_save_path="saved_weights/rel_kde_test_IIIC.pt")
# print("Relevance Density Estimated!")
# cal = ConformalCovariateCalibrator(interpreter, method="softmax", kde_cal=kde_cal, kde_test=kde_test, cal_loader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/rel_cov_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)


# print("Class Conditional Relevance Covariate Shift Test Run!")
# q_hats = []
# kde_cal = RelevanceDensityEstimator(interpreter, cal_loader, k=1, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_cal_IIIC.pt", use_labels=True)
# kde_test = RelevanceDensityEstimator(interpreter, test_loader, k=1, class_index=None, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_test_IIIC.pt")
# cal = ClassConformalCalibrator(interpreter, method="cov_softmax_quantile", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# q_hats = np.array(q_hats).transpose()
# print("Time:", time.time() - start)
# np.savetxt(f"uq/q_hat/IIIC/cls_rel_cov_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")


# # class specific covariate class softmax, just leave it at this.
# print("Class Conditional Relevance With Multiple KDEs")
# q_hats = []
# to_saved_paths_kde_test = {i : f"saved_weights/IIIC_aenc_kde_test_class{i}_nbs{normalize_by_sample()}.pt" for i in range(6)}
# to_saved_paths_kde_cal = {i : f"saved_weights/IIIC_aenc_kde_cal_class{i}_nbs{normalize_by_sample()}.pt" for i in range(6)}
# kde_cals, kde_tests = get_class_relevance_kdes(cal_loader, test_loader, interpreter, kde_dim=8, nEpochs=30, saved_paths_cal=to_saved_paths_kde_cal, saved_paths_test=to_saved_paths_kde_test)
# cal = ClassConformalCovariateCalibrator(interpreter, method="softmax", kde_cals=kde_cals, kde_tests=kde_tests, cal_loader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))
# q_hats = np.array(q_hats).transpose()
# np.savetxt(f"uq/q_hat/IIIC/cls_rel_{cal.method}_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")
# print("Time:", time.time() - start)


# print("Logit Covariate Shift Score Test Run!")
# q_hats = []
# kde_cal = LogitDensityEstimator(interpreter, cal_loader, nEpochs=50)
# kde_test = LogitDensityEstimator(interpreter, test_loader, nEpochs=50)
# print("Logit Density Estimated!")
# cal = ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/score_logit_cov_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)


print("Class Conditional Relevance Covariate Score Shift Test Run!")
q_hats = []
kde_cal = RelevanceDensityEstimator(interpreter, cal_loader, k=1, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_cal_IIIC.pt", use_labels=True)
kde_test = RelevanceDensityEstimator(interpreter, test_loader, k=1, class_index=None, nEpochs=50, kde_dim=6, saved_path="saved_weights/rel_kde_test_IIIC.pt")
cal = ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
start = time.time()
for alpha in alphas:
    q_hat = cal.calibrate(cal_loader, alpha)
    q_hats.append(q_hat.item())
    print("alpha, q_hat, prediction set, prediction set length")
    pred_set = cal.predict_set(signal, q_hat)
    print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

write_to_csv(f"uq/q_hat/IIIC/score_rel_cov_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)
print("Time:", time.time() - start)



# want to compare 3 types fo class conformal calibrators

# # base softmax
# print("Class Conditioned Softmax")
# cal = ClassConformalCalibrator(interpreter, method="softmax", kde_cal=None, kde_test=None, cal_dataloader=cal_loader)
# start = time.time()
# q_hats = []
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# q_hats = np.array(q_hats).transpose()
# print("Time:", time.time() - start)
# np.savetxt(f"uq/q_hat/IIIC/class_{cal.method}_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")


# general covariate class softmax
# print("Class Conditioned Covariate Softmax")
# kde_cal = AttentionDensityEstimator(interpreter, cal_loader, nEpochs=50, saved_path="saved_weights/kde_cal_IIIC.pt")
# kde_test = AttentionDensityEstimator(interpreter, test_loader, nEpochs=50, saved_path="saved_weights/kde_test_IIIC.pt")
# cal = ClassConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# start = time.time()
# q_hats = []
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))
# q_hats = np.array(q_hats).transpose()
# np.savetxt(f"uq/q_hat/IIIC/class_{cal.method}_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")

# print("Time:", time.time() - start)

# # MLP class softmax
# print("Class Conditioned Conformal Covariate Softmax")
# kde_cal = MLPDensityEstimator(interpreter, cal_loader, nEpochs=5, to_save_path="saved_weights/kde_mlp_cal_IIIC.pt")
# kde_test = MLPDensityEstimator(interpreter, test_loader, nEpochs=5, to_save_path="saved_weights/kde_mlp_test_IIIC.pt")
# cal = ClassConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# start = time.time()
# q_hats = []
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(qhat_dict_to_vector(q_hat))
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))
# q_hats = np.array(q_hats).transpose()
# np.savetxt(f"uq/q_hat/IIIC/class_mlp_{cal.method}_nbs{normalize_by_sample()}.csv", q_hats, delimiter=",")

# print("Time:", time.time() - start)


# q_hats = []
# # we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
# print("ROAD")
# cal = ConformalCalibrator(interpreter, method="ROAD")
# start = time.time()
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)
#     q_hats.append(q_hat.item())
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# print("Time:", time.time() - start)
# write_to_csv(f"uq/q_hat/IIIC/ROAD_nbs{normalize_by_sample()}.csv", alphas, q_hats, column_names=column_names)






# q_hats = []
# start = time.time()
# print("ROAD SCORE")
# cal = ConformalCalibrator(interpreter, method="ROAD")
# for alpha in alphas:
#     q_hat = cal.calibrate(cal_loader, alpha)#get_calibration_cutoff(model=model, 
#     #                     calibration_dataloader=cal_loader,
#     #                         interpreter=interpreter, 
#     #                         cal_scoring_function=cal_road_score, alpha=alpha)
#     # q_hats.append(q_hat)
#     # q_hat = 3.1235
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = cal.predict_set(signal, q_hat) #generate_prediction_set(signal, model, q_hat, interpreter, road_score, class_index=label)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))


# print("Time:", time.time() - start)
# write_to_csv("uq/q_hat/road_more_alpha.csv", alphas, q_hats, column_names=column_names)


# maybe instead of ROAD, we do ROJohn where we just mask it and see what happens or add noisy, or do average computation
# try ROAR?

# load csv into pandas dataframes for q_hats
# softmax_df  = pd.read_csv('uq/q_hat/softmax.csv')
# e_score_df  = pd.read_csv("uq/q_hat/e_score.csv")
# road_df  = pd.read_csv('uq/q_hat/road.csv')
# q_hats = (softmax_df["q_hat"],
#         e_score_df["q_hat"],
#         road_df["q_hat"])
# alphas = softmax_df["alpha"]

# df, ps, smax = get_all_metrics_to_present(alphas, q_hats, test_loader.dataset, model, interpreter)
# plot_all_metrics(save_prefix="uq/metrics/tuev", cp_eval_df=df, prediction_sets=ps, smax_kl=smax)







# at some point we should also just do default interpretability on top of softmax prediction sets.
# can do that after this experiment hopefully runs successfully.


# do this for just one sample, can do across all later
# test example 


# #  testing for pytorch gradcam metrics
# from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# targets = [ClassifierOutputTarget(label)]
# cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
# norm_signal, cont_mask= interpreter.visualize(signal, label)
# # print(cont_mask.shape)
# cont_mask = cont_mask[np.newaxis,:]
# print(cont_mask.shape)
# # cont_mask = torch.from_numpy(cont_mask)
# scores = cam_metric(signal.cuda(), cont_mask.transpose(), targets, model)
# print(scores)
