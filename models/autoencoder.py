import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        # Decoder layers
        self.decoder = nn.Linear(encoding_dim, input_dim)
        
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        
        # Decoding
        reconstructed = self.decoder(encoded)
        
        return reconstructed
    

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(SparseAutoEncoder, self).__init__()
        
        # Encoder Layer (i.e the dictionary)
        self.dictionary = nn.Linear(input_dim, encoding_dim)
        self.ReLu = nn.ReLU()
       
    def get_learned_dictionary(self):
        return self.dictionary.weight # no need for bias
    # Encode with the encoder
    def encode(self, x):
        encoded = self.ReLu(self.dictionary(x))
        return encoded
    
    # Decode with the encoder
    def decode(self, x):
        return torch.matmul(x, self.dictionary.weight)
    
    def forward(self, x):
        # Encoding
        encoded = self.encode(x)

        # Decoding
        reconstructed = self.decode(encoded)
        
        return reconstructed
    

# we will just assume that everything is on cuda if possible
# input dimension is of the 
def train_sparse_autoencoder(dataloader, input_dim, encoding_dim=8, alpha=4,  num_epochs = 20, save_path=None, device=torch.device('cpu')) -> SparseAutoEncoder:      
    # encoding_dim = 10  # Needs to be a small enough dimension for proper use
    model = SparseAutoEncoder(input_dim, encoding_dim).to(device)
    model.train()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_data in dataloader:
            inputs = batch_data.to(device)  # Assuming your dataset returns a tuple with input data and labels
            # ignore the labels as we won't have any
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, inputs)  # MSE loss

            # Get L1 norm of the sparsity loss
            loss += alpha * torch.norm(model.encode(inputs), p=1) 
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    print("Training complete")
    return model # return the trained model


def dictionary_learn(batch, input_dim, encoding_dim=8, alpha=4,  num_epochs = 20, save_path=None, device=torch.device('cpu')) -> SparseAutoEncoder:      
    # encoding_dim = 10  # Needs to be a small enough dimension for proper use
    model = SparseAutoEncoder(input_dim, encoding_dim).to(device)
    model.train()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
       
        # ignore the labels as we won't have any
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch.to(device))

        # Compute the loss
        loss = criterion(outputs, batch.to(device))  # MSE loss

        # Get L1 norm of the sparsity loss
        loss += alpha * torch.norm(model.encode(batch.to(device)), p=1) 
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / batch.size()[0]}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    print("Training complete")
    return model # return the trained model




def train_autoencoder(dataloader, input_dim, num_epochs=20, device=torch.device("cpu"), encoding_dim=8, save_path=None) -> AutoEncoder:
    model = AutoEncoder(input_dim, encoding_dim).to(device)
    model.train()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_data in dataloader:
            inputs = batch_data.to(device)  # Assuming your dataset returns a tuple with input data and labels
            # ignore the labels as we won't have any
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, inputs)  # MSE loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}")

    print("Training complete")
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model # return the trained model

