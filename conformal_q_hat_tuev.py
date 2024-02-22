import torch
import torch
import csv 
import time
import gc
from models import *
from interpret.chefer import *
from models.st_transformer import *
from models.pytorch_lightning import *
from data import *
from uq.conformal import *


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
depth = 4 
dropout = 0.5
num_heads = 8
model = STTransformer(emb_size=emb_size, 
                                   depth=depth,
                                   n_classes=6, 
                                   channel_length=1000,
                                   dropout=dropout, 
                                   num_heads=num_heads,
                                   kernel_size=11, 
                                   stride=11,
                                   kernel_size2=5,
                                   stride2=5)

model.load_state_dict(torch.load("saved_weights/st_transformer_conformal_tuev.pt"))


# my training hyperparameters
sampling_rate = 200
batch_size = 500 # may be too memory intensive depending on the conformity score...
num_workers = 32
train_loader, test_loader, val_loader, cal_loader = prepare_TUEV_cal_dataloader(sampling_rate=sampling_rate, batch_size=batch_size, num_workers=num_workers)

# get interpreter
interpreter = STTransformerInterpreter(model=model)


alphas = [0.01, 0.05, 0.1, 0.15] 
column_names = ["alpha", "q_hat"]
signal, label = test_loader.dataset[0]
signal = signal.unsqueeze(0)
current_gpu = torch.cuda.current_device()
print("Currently active GPU:", current_gpu)
q_hats = []
print("GradCam++ Softmax Drop")
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache() # empty the cache before iterating

cal = ConformalCalibrator(interpreter,method="softmax_drop")
start = time.time()
for alpha in alphas:
    q_hat = cal.calibrate(cal_loader, alpha)
    q_hats.append(q_hat.item())
    print("alpha, q_hat, prediction set, prediction set length")
    pred_set = cal.predict_set(signal, q_hat)
    print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))


print("Time:", time.time() - start)
write_to_csv("uq/q_hat/tuev/softmax_drop.csv", alphas, q_hats, column_names=column_names)
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache() # empty the cache before iterating

q_hats = []
# we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
print("SOFTMAX")
cal = ConformalCalibrator(interpreter, method="softmax")
start = time.time()
for alpha in alphas:
    q_hat = cal.calibrate(cal_loader, alpha)
    q_hats.append(q_hat.item())
    print("alpha, q_hat, prediction set, prediction set length")
    pred_set = cal.predict_set(signal, q_hat)
    print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))
    # all_pred_sets, acc, mean_pred_length , set_lengths = get_prediction_set_metrics(cal_loader, cal, q_hat)
    # print("acc:", acc)
    # print("mean pred set length:", mean_pred_length)

print("Time:", time.time() - start)
write_to_csv("uq/q_hat/tuev/softmax.csv", alphas, q_hats, column_names=column_names)
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache() # empty the cache before iterating

q_hats = []
# we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
print("MORF")
cal = ConformalCalibrator(interpreter, method="MORF")
start = time.time()
for alpha in alphas:
    q_hat = cal.calibrate(cal_loader, alpha)
    q_hats.append(q_hat.item())
    print("alpha, q_hat, prediction set, prediction set length")
    pred_set = cal.predict_set(signal, q_hat)
    print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

print("Time:", time.time() - start)
write_to_csv("uq/q_hat/tuev/MORF.csv", alphas, q_hats, column_names=column_names)



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
# write_to_csv("uq/q_hat/tuev/ROAD.csv", alphas, q_hats, column_names=column_names)






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
