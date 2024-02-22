# returns a pandas dataframe of 
# let q_hats be a tuple of arrays
# "alphas, softmax avg ps len, e_score avg ps len, road avg ps len, softmax acc, e_score acc, road acc, "
# also return a vector of pairwise kl_divergence
# will return all scoring functions 
# def get_all_metrics_to_present(alphas, q_hats, dataset, model, interpreter):
#     column_names = ["Alpha, Softmax Avg PS Length, E_Score Avg PS Length, ROAD Avg PS Length, Softmax Acc, E_score Acc, ROAD Acc"]
#     softmax_q, e_q, road_q = q_hats 

#     softmax_ps = []
#     e_ps = []
#     road_ps = []

#     softmax_acc = [] # does the guarantee hold (?), sanity check
#     e_acc = []
#     road_acc = []

#     softmax_ps_len = []
#     e_ps_len = []
#     road_ps_len = []
#     # softmax
#     for i in range(len(alphas)):
#         q_hat = softmax_q[i]
#         ps, acc, avg_ps_len = get_prediction_set_metrics(dataset=dataset, model=model, q_hat=q_hat, 
#                                    interpreter=interpreter, scoring_function=softmax)
#         softmax_ps.append(ps)
#         softmax_acc.append(acc)
#         softmax_ps_len.append(avg_ps_len)

#     #e_score
#     for i in range(len(alphas)):
#         q_hat = e_q[i]
#         ps, acc, avg_ps_len = get_prediction_set_metrics(dataset=dataset, model=model, q_hat=q_hat, 
#                                    interpreter=interpreter, scoring_function=e_score)
#         e_ps.append(ps)
#         e_acc.append(acc)
#         e_ps_len.append(avg_ps_len)

#     # road
#     for i in range(len(alphas)):
#         q_hat = road_q[i]
#         ps, acc, avg_ps_len = get_prediction_set_metrics(dataset=dataset, model=model, q_hat=q_hat, 
#                                    interpreter=interpreter, scoring_function=road_score)
#         road_ps.append(ps)
#         road_acc.append(acc)
#         road_ps_len.append(avg_ps_len)
    
#     data = {column_names[0] : alphas, 
#             column_names[1] : softmax_ps_len,
#             column_names[2] : e_ps_len,
#             column_names[3] : road_ps_len,
#             column_names[4] : softmax_acc,
#             column_names[5] : e_acc,
#             column_names[6] : road_acc
#             }
#     df = pd.DataFrame(data)

#     # just compute pairwise differences between softmax and everyone else
#     # also return each of the values for histogramming
#     kl_smax_e = []
#     kl_smax_road = []
#     for i in range(len(alphas)):
#         kl = compute_KL_distance(softmax_ps[i], e_ps[i])
#         kl_road = compute_KL_distance(softmax_ps[i], road_ps[i])
#         kl_smax_e.append(kl)
#         kl_smax_road.append(kl_road)

#     return df, (softmax_ps, e_ps, road_ps), (kl_smax_e, kl_smax_road)


# def plot_all_metrics(save_prefix, cp_eval_df, prediction_sets, smax_kl):

#     # plot line chart for each alpha
#     plt.figure()
#     alphas = cp_eval_df["Alpha"]
#     n_alpha = len(alphas)
#     column_names = cp_eval_df.columns 
#     for cname in column_names:
#         if cname != "Alpha":
#             plt.plot(alphas, cp_eval_df[cname], label=cname)
#     plt.savefig(save_prefix + "_cp_evaluation.png")

#     # plot histograms for prediction sets
#     softmax_ps, e_ps, road_ps = prediction_sets
#     plt.figure()
#     # make subplots n_alpha x 3
#     fig, axes = plt.subplots(5, figsize=(12,10))
#     for i in range(n_alpha):
#         axes[i].hist(softmax_ps[i], label="Softmax")
#         axes[i].hist(e_ps[i], label="E_Score", alpha=0.3)
#         axes[i].hist(road_ps[i], label="ROAD", alpha=0.3)
#     plt.savefig(save_prefix + "_prediction_sets.png")
#     # plot bar charts for KL divergence (Later)
#     print("KL Div Softmax vs. E, Softmax vs. ROAD")
#     print(smax_kl)

#     return None


# # This is arguably a better useful explainability score
# STUFF TO REMOVE EVENTUALLY


# def softmax(input, model, interpreter=None):
#     model = model.cuda()
#     return model(input.cuda()).softmax(1).squeeze()



# def cal_softmax(input, model, interpreter=None, class_index=None):
#     model = model.cuda()
#     return model(input.cuda()).softmax(1).squeeze().detach().cpu().numpy()[class_index]

# #
# def road_score(input, model, interpreter, class_index=None):
#     if len(input.size()) < 3:
#         input = input.unsqueeze(0)
#     scores = []
#     # compute targets repeatedly 
#     for i in range(6):
#         targets = [ClassifierOutputTarget(i)] # go through each plausible class
#         cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
#         norm_signal, cont_mask= interpreter.visualize(input, i, show=False)
#         cont_mask = cont_mask[np.newaxis,:]
#         scores.append(cam_metric(input.cuda(), cont_mask.transpose(), targets, model))    
#     # print(cont_mask.shape)
#     scores = np.array(scores)
#     # print(cont_mask.shape)
#     # cont_mask = torch.from_numpy(cont_mask)
#     scores = scores.reshape(-1)
#     scores = torch.from_numpy(scores)
#     # print(scores)
#     return scores

# def cal_road_score(input, model, interpreter, class_index=None):
#     if len(input.size()) < 3:
#         input = input.unsqueeze(0)   
#     # compute targets repeatedly 

#     targets = [ClassifierOutputTarget(class_index)]
#     cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
#     norm_signal, cont_mask= interpreter.visualize(input, class_index, method="linear", show=False)
#     # print(cont_mask.shape)
#     cont_mask = cont_mask[np.newaxis,:]
#     # print(cont_mask.shape)
#     # cont_mask = torch.from_numpy(cont_mask)
#     scores = cam_metric(input.cuda(), cont_mask.transpose(), targets, model)
#     scores = scores.reshape(-1)
#     return scores


# # pitfall is one could still in principle submit a terrible cam * input map and get higher confidence and still do well.
# # for no calibration
# def e_score(input, model, interpreter, class_index = None):
#     og_softmax = 0
#     with torch.no_grad():
#         og_softmax = interpreter.model(input.cuda()).softmax(1)
#     cam = interpreter.get_cam_on_image(input, class_index = class_index, method='linear')
#     interpreted_softmax = model(cam.cuda()).softmax(1)
#     score = og_softmax - interpreted_softmax
#     # print(score)
#     return -score # get the difference values for each class

# def cal_e_score(input, model, interpreter, class_index = None):
#     # get original softmax score
#     default_softmax = 0
#     with torch.no_grad():
#         default_softmax = interpreter.model(input.cuda()).softmax(1)
#     if class_index == None:
#         class_index = np.argmax(default_softmax.cpu().data.numpy(), axis=-1)
#     # get cam inputted softmax score
#     cam_input = interpreter.get_cam_on_image(input, class_index = class_index, method="linear")
#     # np_img, cam_input = interpreter.visualize(input, class_index=class_index, show=False)
#     # cam_input = cam_input * np_img
    
#     interpreted_softmax = model(cam_input.cuda()).softmax(1).detach()
#     # print(interpreted_softmax)
#     # compute some differnence for the true class difference
#     score = default_softmax - interpreted_softmax     # smaller difference == greater conformity
  
#     score = score.squeeze()
#     score = score.detach().cpu().numpy()[class_index]

#     # take reverse such that greater difference == greater conformity
#     return -score

# # instead of doing argmax or true label, we instead want to do the sum of 
# # all sorted scores up to the true label
# def cal_cumulative_e_score(input, model, interpreter, class_index=None):
#     model = model.cuda()
#     # get original softmax score
#     default_softmax = F.softmax(model(input.cuda()))

#     # use max if nothing specified, hence make sure to specify! when using this
#     # get cam inputted softmax score
#     cam_input = interpreter.get_cam_on_image(input, class_index = class_index, method="linear")
#     interpreted_softmax = F.softmax(model(cam_input.cuda()))

#     #  get cumulative difference until true class or class specified. 
#     interpreted_softmax = default_softmax - interpreted_softmax

#     sorted_int_softmax, indices = torch.sort(interpreted_softmax, descending=True)
#     score = 0
#     for i in indices:
#         if i != class_index:
#             score += interpreted_softmax

#     # smaller difference == greater conformity
#     # take reverse such that greater value == greater conformity
#     return -score # maybe? read more




# once done training, can use this with respect to some level of true dataset
# def get_calibration_cutoff(model, calibration_dataloader, interpreter, cal_scoring_function, alpha=0.1):

#     # compute sorted calibration scores 
#     e_scores = []
#     for signal, label in calibration_dataloader:
#         e_score = cal_scoring_function(input=signal, model=model, interpreter=interpreter, class_index = label) # should return a scalar
#         # torch.cuda.empty_cache() # empty the cache after each iteration
#         # print(e_score)
#         e_scores.append(e_score)

#     e_scores = np.array(e_scores)
#     # sort
#     e_scores.sort()

#     # use the conformal prediction equation to calculate q_hat score
#     # get quantile defined by alpha
#     n = len(calibration_dataloader.dataset)
#     # questions to ask, i.e
#     # print("Conformal Calibration Complete!")
#     # print("cal_average:", e_scores.mean())
#     # print("cal_variance:", e_scores.var())
#     # print("cal_median:", np.median(e_scores)) # get an idea of how skewed it is
#     # print("cal_shape:", e_scores.shape)
#     q_level = np.ceil((n+1) * (1-alpha)) / n 
#     q_level = 1 - q_level # for conformity score
#     q_hat = np.quantile(e_scores, q_level, method='higher')
#     return q_hat


# # in principle, now we can simply gear our prediction sets based off of an explicit class_index on what we think it is.
# def generate_prediction_set(input, model, q_hat, interpreter, scoring_function ,class_index = None):
#     scores = scoring_function(input, model,interpreter)
#     # sort
#     sorted_scores, indices = torch.sort(scores)
#     # print(sorted_scores.size())
#     indices = indices.squeeze()
#     sorted_scores = sorted_scores.squeeze()
#     # generate prediction set
#     # print(sorted_scores)
#     prediction_set = []
#     for i in range(sorted_scores.size()[0]):
#         if sorted_scores[i] > q_hat:
#             prediction_set.append(indices[i].detach().cpu().numpy()) # returns a torch 
    
#     return prediction_set


    # should b the slowest just bc not actually on gpu.
    # we will do a rewrite later once we've finished reading the paper entirely and looked through thier code!
# class ROAD(InterpretableCalibrationScore):
#     def cal_score(self, input, class_index):
        
#         if len(input.size()) < 3:
#             input = input.unsqueeze(0) # for the batch space
    
#         cont_mask= self.interpreter.get_cam(input, class_index, method="linear") # get all cams w/r to class_index in parallel
      
#         # now compute road_scores across entire_batch (Note: we will have to rewrite this at some point to be parallelized)
#         scores = np.array([])
#         for i in range(input.size()[0]):
#             targets = [ClassifierOutputTarget(class_index[i])]
#             cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
        
#             # input is C x H x W
#             # masks is nClassMasks x W x H , where nClassMasks are different masks of w/r diff class for same input(I believe)...
#             score = cam_metric(input[i].to(self.interpreter.device).unsqueeze(0), 
#                             cont_mask[i].unsqueeze(0).detach().cpu().numpy().transpose(),
#                             targets, self.interpreter.model).reshape(-1)
#             scores = np.concatenate((scores, score), axis=0)
#         scores = scores.reshape(-1)
#         return torch.from_numpy(scores).to(self.interpreter.device).detach()
    
#     def score(self, input):
#         if len(input.size()) < 3:
#             input = input.unsqueeze(0)
#         scores = []
#         # compute targets repeatedly 
#         for i in range(6):
#             targets = [ClassifierOutputTarget(i)] # go through each plausible class
#             cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
#             cont_mask= self.interpreter.get_cam(input, i)
#             # cont_mask = cont_mask[np.newaxis,:]
#             scores.append(cam_metric(input.to(self.interpreter.device), 
#                                      cont_mask.unsqueeze(0).detach().cpu().numpy().transpose(), 
#                                      targets, self.interpreter.model))    
#         # print(cont_mask.shape)
#         scores = np.array(scores)
#         # print(cont_mask.shape)
#         # cont_mask = torch.from_numpy(cont_mask)
#         scores = scores.reshape(-1)
#         scores = torch.from_numpy(scores)
#         # print(scores)
#         return scores.to(self.interpreter.device).detach()
