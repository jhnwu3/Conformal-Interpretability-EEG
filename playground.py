import scipy.stats 
import numpy as np
import torch
from data import *
from models.autoencoder import *
from interpret.chefer import *
from einops import rearrange
from uq.kde import *
from uq.covariate import *
from uq.conformal import *
from uq.class_conditional import *
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
num_workers = 32
train_loader, test_loader, val_loader, cal_loader = prepare_IIIC_cal_dataloader(batch_size=batch_size, num_workers=num_workers, sample_norm=normalize_by_sample())
# get interpreter
interpreter = STTransformerInterpreter(model=model)

print("Performing Density Estimation!") # we should find out if this just takes a long time
# try out the density estimator
k = 1


# 6 for the number of classes
to_saved_paths_kde_test = {i : f"saved_weights/IIIC_aenc_kde_test_class{i}" for i in range(6)}
to_saved_paths_kde_cal = {i : f"saved_weights/IIIC_aenc_kde__cal_class{i}" for i in range(6)}

kde_cals, kde_tests = get_class_relevance_kdes(cal_loader, test_loader, interpreter)


# kde_cals = {1:"garbage"}
# kde_tests = {1:"garbage"}
cc = ClassConformalCovariateCalibrator(interpreter, method="softmax", kde_cals=kde_cals, kde_tests=kde_tests)
cc.calibrate(cal_loader, alpha=0.1)


# cc = ClassConformalCalibrator(interpreter, method="softmax", cal_dataloader=cal_loader)
# q_hats = cc.calibrate(cal_loader, alpha=0.1)
# print(q_hats)
batch, label = next(iter(test_loader))
# bs qhats
pred_set = cc.predict_set(batch, q_hats={0:0.01, 1:0.01, 2:0.01, 3:0.01, 4:0.01, 5:0.01})
print(pred_set.size())

# kde_cal = InterpretableDensityEstimator(interpreter, cal_loader,k=3, nEpochs=50, to_save_path=f"saved_weights/kde_cal_IIIC_k{k}.pt")
# kde_test = InterpretableDensityEstimator(interpreter, test_loader,k=3, nEpochs=50, to_save_path=f"saved_weights/kde_test_IIIC_k{k}.pt")

# # try out the covariate shift weighed softmax
# cp = ConformalCalibrator(interpreter, method="cov_softmax", kde_cal=kde_cal, kde_test=kde_test, cal_dataloader=cal_loader)
# q_hat = cp.calibrate(cal_loader, alpha=0.1)
# acc, mean_len, pred_set_lens = get_prediction_set_metrics(test_loader, cp, q_hat=q_hat)
# print("acc:", acc)
# print('mean pred set len:', mean_len)
# print("q_hat:", q_hat)
# plt.figure()
# plt.title("Covariate Shift Softmax Prediction Set Lengths")
# plt.hist(pred_set_lens, bins=6)
# plt.ylabel("# of Samples")
# plt.xlabel("Prediction Set Length")
# plt.savefig(f"fig/cp/IIIC/covariate_shift_softmax_pred_set_lens_k{k}.png")

