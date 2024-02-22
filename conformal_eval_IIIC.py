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
from graph import *
# my training hyperparameters

batch_size = 276
num_workers = 4
lr = 0.01
total_epochs = 50

# my model hyperparameters
emb_size = 256
depth = 6 
dropout = 0.5
num_heads = 8
patch_kernel_length = 11  # cqi = 15 - UNUSED
stride = 11  # cqi = 8 - UNUSED

train_loader, test_loader, val_loader, cal_loader = prepare_IIIC_cal_dataloader(batch_size=batch_size, num_workers=num_workers, drop_last=False, sample_norm=normalize_by_sample())
signal, label = train_loader.dataset[0]
# print(signal)
# exit(0)
# define the model for training - STT transformer
st_transformer = STTransformer(emb_size=emb_size, 
                                depth=depth,
                                n_classes=6, 
                                channel_length=2000,
                                dropout=dropout, 
                                num_heads=num_heads,
                                kernel_size=11, 
                                stride=11,
                                kernel_size2=11,
                                stride2=11)

st_transformer.load_state_dict(torch.load(f"saved_weights/st_transformer_conformal_IIIC_nbs{normalize_by_sample()}.pt"))
st_transformer = st_transformer.cuda()



# get interpreter
interpreter = STTransformerInterpreter(model=st_transformer)
get_plots_research(test_loader, cal_loader, interpreter, dataset='IIIC', normalize=normalize_by_sample())
# plot_stored_data()



# cal = ConformalCalibrator(interpreter, method="softmax")
# validator = InterpretableWrapper(calibrator=cal,interpretable_scorer=MORF(interpreter))
# # 0.001177854253910482
# acc_hard, acc_easy, acc_hard_in = validate_interpretable(test_loader, validator=validator, q_hat=7.848030918466975e-07)
# print("acc of prediction sets greater than 1 with highest interpretability score == label:", acc_hard)
# print("acc of prediction sets == 1:", acc_easy)
# print("acc of pred set > 1, contains ground truth", acc_hard_in)
# # evaluate on test_loader
# y_true = []
# y_prob = []

# for signal, label in test_loader:
#     prob = st_transformer(signal.cuda()).softmax(1)
#     # append every "batch" 
#     for i in range(prob.size()[0]):
#         y_true.append(label[i])
#         y_prob.append(prob[i].detach().cpu().numpy())

# y_true = np.array(y_true)
# y_prob = np.array(y_prob)

# print(y_true.shape)
# print(y_prob.shape)
# print(multiclass_metrics_fn(y_true, y_prob, metrics=["accuracy", "roc_auc_macro_ovr"]))

# calibration step after validation and what not.

# now do with the conformity scores



# train model with pytlightning
