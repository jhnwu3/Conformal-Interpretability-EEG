import scipy.stats 
import numpy as np
import torch
import torch.utils.data
import gc
from data import *
from models.autoencoder import *
from interpret.chefer import *
from uq.covariate import *
from einops import rearrange

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # normalize each sample 
        return self.X[index]

class ClassSpecificDataset(torch.utils.data.Dataset):
    def __init__(self, X, class_index= None):
        self.X = X
        self.class_index = class_index
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # normalize each sample 
        return self.X[index], self.class_index

class AvgAttentionTopKDataset(torch.utils.data.Dataset): # type: ignore
    # get top k = 1
    # Values should be N x NT x D
    # Attention should also be N x NT  
    # where D is the embedding dimension
    # NT is the number of tokens
    def __init__(self, avg_attentions, avg_values, k=1):
        self.attentions = avg_attentions # can be relevance as well!
        self.values = avg_values 
        # get top k
        self.attentions, self.indices = torch.topk(self.attentions, k=k, dim=-1)
        # get the corresponding values
        self.values = torch.gather(self.values, dim=1, index=self.indices.unsqueeze(-1).expand(-1, -1, self.values.size()[-1])).reshape(self.values.size()[0], -1)
    def __len__(self):
        return self.values.size()[0]
    
    def __getitem__(self, idx):
        return self.values[idx]

def get_relevance_kdes(cal_loader, test_loader, interpreter, k=1, nEpochs=1, kde_dim=8, saved_path=None, to_save_path=None):
    
    
    cal_kde = RelevanceDensityEstimator(interpreter, cal_loader, k=k, nEpochs=nEpochs, kde_dim=kde_dim, saved_path=saved_path, to_save_path=to_save_path, use_labels=True)
    test_kde = RelevanceDensityEstimator(interpreter, test_loader, k=k, class_index=None, nEpochs=nEpochs, kde_dim=kde_dim, saved_path=saved_path, to_save_path=to_save_path)
    
    return cal_kde, test_kde


# should return 2 dictionaries, one for calibration and one for test where the key is the class index
def get_class_relevance_kdes(cal_loader, test_loader, interpreter, k=1, nEpochs=1, kde_dim=8,
                             saved_paths_cal=None, saved_paths_test = None, to_save_paths_cal=None, to_save_paths_test=None, 
                             n_classes=6):
    
    

    cal_kdes = {}
    test_kdes = {}
    # calibrate for each class 
    all_labels = []
    all_signals = []
    for signal, label in cal_loader:
        all_labels.append(label)
        all_signals.append(signal)
        gc.collect()
    
        with torch.no_grad():
            torch.cuda.empty_cache() # empty the cache before iterating
    
    all_labels = torch.cat(all_labels)
    all_signals = torch.cat(all_signals)
    unique_labels = torch.unique(all_labels)
    class_signals = {label.item(): all_signals[all_labels == label] for label in unique_labels}
    # saved_paths = {label.item(): None for label in unique_labels}
    if saved_paths_cal == None:
        saved_paths_cal = {key.item(): None for key in unique_labels}
    
    if saved_paths_test == None: 
        saved_paths_test = {key.item(): None for key in unique_labels}

    if to_save_paths_cal == None:
        to_save_paths_cal = {key.item(): None for key in unique_labels}
    if to_save_paths_test == None: 
        to_save_paths_test = {key.item(): None for key in unique_labels}

    for key, signals in class_signals.items():
        dataloader = torch.utils.data.DataLoader(ClassSpecificDataset(signals, key), batch_size=1600, shuffle=True, num_workers=32)
        cal_kdes[key] = RelevanceDensityEstimator(interpreter, dataloader, k=k, kde_dim=kde_dim, nEpochs=nEpochs, class_index=torch.tensor(key), saved_path=saved_paths_cal[key], to_save_path=to_save_paths_cal[key])

    # transductive setting, DO NOT TOUCH THE LABELS, just use the inductive bias of the transformer i.e use the argmax'd relevance
    # to create some pseudo test distributions (that may or may not look different!)

    test_kdes = get_test_kdes(interpreter, test_loader, k=k, nEpochs=nEpochs, kde_dim=kde_dim, saved_paths=saved_paths_test, to_save_paths=to_save_paths_test)

    # for label in unique_labels:
    #     # this will be much larger, so might take substantially more time! Literally just compute P(x|y_hat) for each class
    #     # will need to reiterate using top classes instead.
    #     test_kdes[label.item()] = RelevanceDensityEstimator(interpreter, test_loader, k=k, nEpochs=nEpochs, class_index=label, saved_path=saved_paths_test[label.item()], to_save_path=to_save_paths_test[label.item()])

    return cal_kdes, test_kdes




def get_test_kdes(interpreter :STTransformerInterpreter, dataloader, k=1, nEpochs=1, kde_dim=8, nTop=1, saved_paths=None, to_save_paths=None):
    # self.class_index = class_index
    # concatenate all batches of attention of the last layer
    test_kdes = {}
    relevances = []
    class_labels = []
    values = []
    for batch, label in dataloader:
        softmax = interpreter.model(batch)
        # sort the softmax scores
        value, indices = torch.sort(softmax, dim=-1, descending=True)
        # top 2 classes predicted
        value = interpreter.get_average_values(batch)
        for i in range(nTop):
            # print(indices[:,i].shape)
            # print(indices[:,i])
            relevances.append(interpreter.get_relevance_matrix(batch, indices[:,i]))
            class_labels.append(indices[:,i])
            values.append(value) # lets just assume we have enough memory for now!
            # rel2 = self.interpreter.get_relevance_matrix(batch, indices[:,1]) 
        
        
    # we should return a dataset for each class?
    relevances = torch.cat(relevances)
    values = torch.cat(values)
    class_labels = torch.cat(class_labels)
    

    print(relevances.shape)
    print(values.shape)
    
    # now that we have all of them we can create a dataset for each class
    unique_classes = torch.unique(class_labels)
    relevances_per_class = {label.item(): relevances[class_labels == label] for label in unique_classes}
    values_per_class = {label.item(): values[class_labels == label] for label in unique_classes}
    # datasets = {label.item(): ClassSpecificDataset(relevances_per_class[label.item()], values_per_class[label.item()]) for label in unique_classes}
    
    # create dataloaders for each class to train multiple autoencoders
    # dataloaders = {label.item(): torch.utils.data.DataLoader(datasets[label.item()], batch_size=500, shuffle=True, num_workers=1) for label in unique_classes}
    if saved_paths == None: 
        saved_paths = {key.item(): None for key in unique_classes}    
    if to_save_paths == None: 
        to_save_paths = {key.item(): None for key in unique_classes}

    for label, relevances in relevances_per_class.items():
        print("label:", label)
        test_kdes[label] = TestRelevanceDensityEstimator(interpreter, relevances=relevances, values=values_per_class[label], k=k, nEpochs=nEpochs, kde_dim=kde_dim, class_index=label, saved_path=saved_paths[label], to_save_path=to_save_paths[label])
    return test_kdes


class TestRelevanceDensityEstimator():
    def __init__(self, interpreter :STTransformerInterpreter, relevances, values, k=1, nEpochs=1, kde_dim=8, class_index=None, saved_path=None, to_save_path=None):
        self.interpreter = interpreter
        self.k = k
        self.class_index = torch.tensor(class_index)
        # concatenate all batches of attention of the last layer
        self.dataset = AvgAttentionTopKDataset(relevances, values, self.k)
        
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=2000, shuffle=True, num_workers=24)
        self.aenc = AutoEncoder(values.size()[-1], kde_dim).to(self.interpreter.device) # 8 due to dataset size
        if saved_path is not None:
            self.aenc.load_state_dict(torch.load(saved_path))
        else:
            # train autoencoder, give it a validation dataset so we can stop early for autoencoder
            self.aenc = train_autoencoder(self.dataloader, input_dim=self.dataset[0].size()[-1], encoding_dim=8,
                                           num_epochs=nEpochs, device=self.interpreter.device, 
                                           save_path=to_save_path)       
        
        self.aenc.eval()
        # now we can convert entire dataset into latent space
        latent_space = []
        for batch in self.dataloader:
            latent_space.append(self.aenc.encode(batch))

        latent_space = torch.cat(latent_space)
        print(latent_space.shape)
        # convert to numpy
        self.estimator = scipy.stats.gaussian_kde(latent_space.detach().cpu().numpy().T)
    
    # should return multiple probabilities
    def __call__(self, data):
        attn = self.interpreter.get_relevance_matrix(data, self.class_index)
        values = self.interpreter.get_average_values(data)
        attn, indices = torch.topk(attn, k=self.k, dim=-1)
        values = torch.gather(values, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, values.size()[-1])).reshape(values.size()[0], -1)
        return self.estimator(self.aenc.encode(values).detach().cpu().numpy().T)


# for the test estimator we need to get the top k classes for each sample, and then compute the density for each class
# i.e we should return k density estimators that averages into one class estimator and return that instead.
# In principle, this should cover more of the space 
class RelevanceDensityEstimator():
    def __init__(self, interpreter :STTransformerInterpreter, dataloader, k=1, nEpochs=1, kde_dim=8, class_index=None, saved_path=None, to_save_path=None, use_labels=False):
        self.interpreter = interpreter
        self.k = k
        self.class_index = class_index
        # concatenate all batches of attention of the last layer
        attentions = []
        values = []
        for batch, label in dataloader:
            attn = None
            if use_labels:
                attn = self.interpreter.get_relevance_matrix(batch, label) # bad code again lol
            else:
                attn = self.interpreter.get_relevance_matrix(batch, class_index)

            value = self.interpreter.get_average_values(batch)
            attentions.append(attn)
            values.append(value)
        
        attentions = torch.cat(attentions)
        values = torch.cat(values)
        print(attentions.shape)
        print(values.shape)

        
        self.dataset = AvgAttentionTopKDataset(attentions, values, self.k)
        
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=2000, shuffle=True, num_workers=16)
        self.aenc = AutoEncoder(values.size()[-1], kde_dim).to(self.interpreter.device) # 8 due to dataset size
        if saved_path is not None:
            self.aenc.load_state_dict(torch.load(saved_path))
        else:
            # train autoencoder, give it a validation dataset so we can stop early for autoencoder
            self.aenc = train_autoencoder(self.dataloader, input_dim=self.dataset[0].size()[-1], encoding_dim=kde_dim,
                                           num_epochs=nEpochs, device=self.interpreter.device, 
                                           save_path=to_save_path)       
        
        self.aenc.eval()
        # now we can convert entire dataset into latent space
        latent_space = []
        for batch in self.dataloader:
            latent_space.append(self.aenc.encode(batch))

        latent_space = torch.cat(latent_space)
        print(latent_space.shape)
        # convert to numpy
        self.estimator = scipy.stats.gaussian_kde(latent_space.detach().cpu().numpy().T)
    
    # should return multiple probabilities
    def __call__(self, data):
        attn = self.interpreter.get_relevance_matrix(data, self.class_index)
        values = self.interpreter.get_average_values(data)
        attn, indices = torch.topk(attn, k=self.k, dim=-1)
        values = torch.gather(values, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, values.size()[-1])).reshape(values.size()[0], -1)
        return self.estimator(self.aenc.encode(values).detach().cpu().numpy().T)



class AttentionDensityEstimator():
    def __init__(self, interpreter : STTransformerInterpreter, dataloader, k=1, nEpochs = 1, saved_path=None, to_save_path=None) -> None:
        self.interpreter = interpreter
        self.k = k
        # concatenate all batches of attention of the last layer
        attentions = []
        values = []
        for batch, label in dataloader:
            attn = self.interpreter.get_average_attentions(batch)
            value = self.interpreter.get_average_values(batch)
            attentions.append(attn)
            values.append(value)
        
        attentions = torch.cat(attentions)
        values = torch.cat(values)
        print(attentions.shape)
        print(values.shape)

        
        self.dataset = AvgAttentionTopKDataset(attentions, values, self.k)
        
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1000, shuffle=True, num_workers=32)
        self.aenc = AutoEncoder(values.size()[-1], 8).to(self.interpreter.device)
        if saved_path is not None:
            self.aenc.load_state_dict(torch.load(saved_path))
        else:
            # train autoencoder, give it a validation dataset so we can stop early for autoencoder
            self.aenc = train_autoencoder(self.dataloader, input_dim=self.dataset[0].size()[-1],
                                           num_epochs=nEpochs, device=self.interpreter.device, 
                                           save_path=to_save_path)       
        
        self.aenc.eval()
        # now we can convert entire dataset into latent space
        latent_space = []
        for batch in self.dataloader:
            latent_space.append(self.aenc.encode(batch))

        latent_space = torch.cat(latent_space)
        print(latent_space.shape)
        # convert to numpy
        self.estimator = scipy.stats.gaussian_kde(latent_space.detach().cpu().numpy().T)

    # data should be a torch tensor
    def __call__(self, data):
        attn = self.interpreter.get_average_attentions(data)
        values = self.interpreter.get_average_values(data)
        attn, indices = torch.topk(attn, k=self.k, dim=-1)
        values = torch.gather(values, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, values.size()[-1])).reshape(values.size()[0], -1)
        return self.estimator(self.aenc.encode(values).detach().cpu().numpy().T)


class MLPDensityEstimator():
    def __init__(self, interpreter, dataloader, saved_path=None, to_save_path=None, nEpochs=1, kde_dim=8):
        self.interpreter = interpreter
        # concatenate all batches of attention of the last layer
        mlp_outs = []
        for batch, label in dataloader:
            activations = self.interpreter.get_avg_mlp_activation(batch) #get_last_mlp_activation(batch)
            # sparse_aenc = dictionary_learn(activations, input_dim=activations.size()[-1],
            #                             num_epochs=nEpochs, device=self.interpreter.device, 
            #                             save_path=to_save_path)  
            
            # print(sparse_aenc.get_learned_dictionary().size())   
            # print(sparse_aenc.encode(activations).size())
            # print(sparse_aenc.encode(activations)[0])
            # exit(0) 
            mlp_outs.append(activations)

        mlp_outs = torch.cat(mlp_outs)
        print(mlp_outs.shape)
        self.dataset = MLPDataset(mlp_outs)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1000, shuffle=True, num_workers=32)
        self.aenc = SparseAutoEncoder(mlp_outs.size()[-1], kde_dim).to(self.interpreter.device)
        if saved_path is not None:
            self.aenc.load_state_dict(torch.load(saved_path))
        else:
            # train autoencoder, give it a validation dataset so we can stop early for autoencoder
            self.aenc = train_autoencoder(self.dataloader, input_dim=self.dataset[0].size()[-1],
                                           num_epochs=nEpochs, encoding_dim=4, device=self.interpreter.device, 
                                           save_path=to_save_path)       

        # Reminder training an autoencoder and a sparse autoencoder globally isn't feasible here!
        self.aenc.eval()
        # now we can convert entire dataset into latent space
        latent_space = []
        for batch in self.dataloader:
            latent_space.append(self.aenc.encode(batch))

        latent_space = torch.cat(latent_space)
        print(latent_space.shape)
        # convert to numpy
        self.estimator = scipy.stats.gaussian_kde(latent_space.detach().cpu().numpy().T)
    
    # the call should extract the necessary feature vector using the trained autoencoder, and then pass it to the density estimator
    def __call__(self, data):
        mlp_out = self.interpreter.get_last_mlp_activation(data)
        latent = self.aenc.encode(mlp_out)
        return self.estimator(latent.detach().cpu().numpy().T)


class LogitDensityEstimator():
    def __init__(self, interpreter, dataloader, saved_path=None, to_save_path=None, nEpochs=1, kde_dim=8):
        self.interpreter = interpreter
        # concatenate all batches of attention of the last layer
        scores = []
        for batch, label in dataloader:
            activations = self.interpreter.model(batch.to(self.interpreter.device)) #get_last_mlp_activation(batch)
            # sparse_aenc = dictionary_learn(activations, input_dim=activations.size()[-1],
            #                             num_epochs=nEpochs, device=self.interpreter.device, 
            #                             save_path=to_save_path)  
            
            # print(sparse_aenc.get_learned_dictionary().size())   
            # print(sparse_aenc.encode(activations).size())
            # print(sparse_aenc.encode(activations)[0])
            # exit(0) 
            scores.append(activations)

        scores = torch.cat(scores)
        print(scores.shape)
        # self.dataset = MLPDataset(mlp_outs)
        # create dataloader
        # self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1000, shuffle=True, num_workers=32)
        # self.aenc = SparseAutoEncoder(mlp_outs.size()[-1], kde_dim).to(self.interpreter.device)
        # if saved_path is not None:
        #     self.aenc.load_state_dict(torch.load(saved_path))
        # else:
        #     # train autoencoder, give it a validation dataset so we can stop early for autoencoder
        #     self.aenc = train_autoencoder(self.dataloader, input_dim=self.dataset[0].size()[-1],
        #                                    num_epochs=nEpochs, encoding_dim=4, device=self.interpreter.device, 
        #                                    save_path=to_save_path)       

        # Reminder training an autoencoder and a sparse autoencoder globally isn't feasible here!
        # self.aenc.eval()
        # # now we can convert entire dataset into latent space
        # latent_space = []
        # for batch in self.dataloader:
        #     latent_space.append(self.aenc.encode(batch))

        # latent_space = torch.cat(latent_space)
        # print(latent_space.shape)
        # convert to numpy
        self.estimator = scipy.stats.gaussian_kde(scores.detach().cpu().numpy().T)
    
    # the call should extract the necessary feature vector using the trained autoencoder, and then pass it to the density estimator
    def __call__(self, data):
        mlp_out = self.interpreter.model(data.to(self.interpreter.device))
        # latent = self.aenc.encode(mlp_out)
        return self.estimator(mlp_out.detach().cpu().numpy().T)

if __name__ == "__main__":
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
    
    # try out the density estimator
    # density_estimator = CalibrationInterpretableDensityEstimator(interpreter, cal_loader)
    