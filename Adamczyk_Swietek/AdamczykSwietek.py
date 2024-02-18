import cv2 as cv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt

IMG_SIZE=128

EPOCHS = 50
batch_size = 32

def save_examplorary_images(X_orig_test, X_phase_test, recovered_output):
    """
    Save exemplary images for visual comparison.

    Parameters:
    - X_orig_test (torch.Tensor): Array containing original images.
    - X_phase_test (torch.Tensor): Array containing phase images.
    - recovered_output (list): Array containing output/recovered images.

    This function takes arrays of original images, phase images, and their recovered outputs. It plots and saves
    exemplary images for visual comparison. For every 100th image up to 500 images, it plots three subplots
    side by side: Original Image, Phase, and Output. It saves the figure as a PNG file with a filename indicating
    the index of the image (e.g., example0.png, example100.png, etc.).
    """
    for i in range(0, 500, 100):
    
        fig=plt.figure(figsize=(15, 5))
        ax = plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(X_orig_test[i][0], cmap='gray', vmin=0, vmax=255)
        
        ax = plt.subplot(132)
        plt.title('Phase')
        plt.imshow(X_phase_test[i][0], cmap='gray', vmin=0, vmax=255)
        
        ax = plt.subplot(133)
        plt.title('Output')
        plt.imshow(recovered_output[i][0], cmap='gray', vmin=0, vmax=255)
        fig.savefig(f"example{i}.png")
        plt.clf()

def train_model(lr, train_loader, val_loader):
    """
    Train an autoencoder model.

    Parameters:
    - lr (float): Learning rate for the optimizer.
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    - val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
    Tuple: Trained encoder and decoder models.

    This function trains an autoencoder model consisting of an encoder and a decoder.
    It iterates through the specified number of epochs, computing training and validation losses
    at each epoch. The models are updated based on the computed losses.

    The encoder and decoder models are initially moved to GPU (if available) and their parameters are set
    to be optimized by the Adam optimizer with the given learning rate. Training and validation losses
    are recorded and saved. After each epoch, if the validation loss decreases or it's the first epoch,
    the models are saved.

    Additionally, the function plots and saves the training and validation loss curves as a JPEG image.
    """
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    parameters = list(encoder.parameters())+ list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    train_losses=[]
    valid_losses=[]

    for epoch in tqdm(range(EPOCHS)):
        encoder.train()
        decoder.train()
        
        train_loss = 0
        for i, (train_orig, train_phase) in enumerate(train_loader):
            orig_image = Variable(train_orig).cuda()
            phase_image = Variable(train_phase).cuda()
            
            optimizer.zero_grad()

            encoder_op = encoder(phase_image)
            decoder_inp = get_decoder_input(encoder_op, orig_image.shape[0])
            output = decoder(encoder_op)
            
            loss=loss_func(output,orig_image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        encoder.eval()
        decoder.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (test_orig, test_phase) in enumerate(val_loader):
                orig_image = Variable(test_orig).cuda()
                phase_image = Variable(test_phase).cuda()

                encoder_op = encoder(phase_image)
                decoder_inp = get_decoder_input(encoder_op, orig_image.shape[0])
                output = decoder(decoder_inp)
                
                loss=loss_func(output,orig_image)
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(val_loader)
        if epoch == 0 or avg_valid_loss < min(valid_losses):
            torch.save([encoder,decoder],f'recover_autoencoder-{lr}.pkl')
        valid_losses.append(avg_valid_loss)
    fig, ax = plt.subplots(2)
    ax[0].plot(train_losses)
    ax[1].plot(valid_losses)
    plt.savefig(f"plot {lr}.jpg")
    plt.clf()

    return (encoder, decoder)

def evaluate_models(models, val_loader):
    """
    Evaluate the performance of trained encoder-decoder models on validation data.

    Parameters:
    - models (list[tuple]): List of tuples containing trained encoder-decoder models.
    - val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
    list: Recovered output images.

    This function evaluates the performance of trained encoder-decoder models on validation data.
    It computes the mean squared error (MSE) loss between the original images and the reconstructed images
    for each model. It iterates through each model and accumulates the total validation loss.

    The function also records the recovered output images for further analysis.
    """
    loss_func = nn.MSELoss()
    valid_loss = 0
    recovered_output=[]
    for encoder, decoder in models:
        with torch.no_grad():
            for orig_image, phase_images in tqdm(val_loader):
                print(phase_images.shape)
                orig_image  = Variable(orig_image).cuda()
                hi_variable = Variable(phase_images).cuda()
                encoder_op = encoder(hi_variable)
                decoder_inp = get_decoder_input(encoder_op, phase_images.shape[0])
                output = decoder(decoder_inp)
                
                loss=loss_func(output,orig_image)
                valid_loss += loss.item()
                output=output.cpu()
                output=output.detach().tolist()
                recovered_output.extend(output)
            
            avg_valid_loss = valid_loss / len(val_loader)
            print(avg_valid_loss, valid_loss)
    
    return recovered_output

def get_dataset(start_id, end_id, img_size):
    """
    Load images from files and preprocess them into a dataset.

    Parameters:
    - start_id (int): Starting ID of the images to load.
    - end_id (int): Ending ID of the images to load (exclusive).
    - img_size (int): Size of the images (assumed square).

    Returns:
    Tuple: Tuple containing two torch Tensors, one for original images and one for phase images.

    This function loads grayscale images and corresponding phase images from files.
    It assumes that the images are named 'image_<ID>.jpg' for original images
    and 'phase_<ID>.jpg' for phase images, where <ID> ranges from start_id to end_id - 1.
    The images are loaded using OpenCV and stored in numpy arrays.

    The function then converts the numpy arrays into torch Tensors and reshapes them
    to have the appropriate dimensions for further processing.
    """
    patch = []
    mask = []
    for i in range(start_id, end_id):
        img = cv.imread(f'data/grayscale/image_{i}.jpg', cv.IMREAD_GRAYSCALE)
        m_img = cv.imread(f'data/phases/image_{i}.jpg', cv.IMREAD_GRAYSCALE)
        patch.append(img)
        mask.append(m_img)
    patch = np.array(patch)
    mask = np.array(mask)
    X_orig=torch.Tensor([patch[i] for i in range(len(patch))])
    X_phase=torch.Tensor([mask[i] for i in range(len(mask))])

    
    X_orig_flat=X_orig.reshape(-1,1,img_size,img_size)
    X_phase_flat=X_phase.reshape(-1,1,img_size,img_size)
    
    return X_orig_flat, X_phase_flat

def get_decoder_input(enc_out, batch_size):
    """
    Prepare the input for the decoder network.

    Parameters:
    - enc_out (torch.Tensor): Output from the encoder network.
    - batch_size (int): Size of the input batch.

    Returns:
    torch.Tensor: Reshaped input tensor for the decoder network.

    This function reshapes the output from the encoder network to prepare it
    as input for the decoder network. It first flattens the output tensor along
    the feature dimensions, then reshapes it to match the expected input shape
    of the decoder network, considering the batch size and image dimensions.
    """
    inp = enc_out.view(batch_size, -1)
    return inp.view(batch_size,IMG_SIZE,64,64)

class Encoder(nn.Module):
    """
    Convolutional neural network encoder module.

    This module defines the architecture for the encoder part of the autoencoder.
    It consists of two convolutional layers followed by batch normalization and ReLU activation.

    Attributes:
    - layer1 (nn.Sequential): Sequential container for the first convolutional layers.
    - layer2 (nn.Sequential): Sequential container for the second convolutional layers.

    Methods:
    - forward(x): Forward pass through the encoder network.
    """
    def __init__(self):
        """
        Initialize the Encoder module.

        This method defines the layers and operations of the encoder network.
        It consists of two convolutional layers followed by batch normalization and ReLU activation.
        """
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),             
                        nn.Conv2d(32,32,3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),
                        nn.ReLU(),
        )
                
    def forward(self,x):
        """
        Forward pass through the encoder network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the encoder layers.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class Decoder(nn.Module):
    """
    Convolutional neural network decoder module.

    This module defines the architecture for the decoder part of the autoencoder.
    It consists of two sets of convolutional transpose layers followed by batch normalization and ReLU activation.

    Attributes:
    - layer1 (nn.Sequential): Sequential container for the first set of convolutional transpose layers.
    - layer2 (nn.Sequential): Sequential container for the second set of convolutional transpose layers.

    Methods:
    - forward(x): Forward pass through the decoder network.

    """
    def __init__(self):
        """
        Initialize the Decoder module.

        This method defines the layers and operations of the decoder network.
        It consists of two sets of convolutional transpose layers followed by batch normalization and ReLU activation.
        """
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(128,128,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),
                        nn.ReLU()
        )
        
    def forward(self,x):
        """
        Forward pass through the decoder network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the decoder layers.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    
class ImageDataset(Dataset):
    """
    Custom PyTorch dataset for images.

    This dataset class is used to create a custom dataset for pairs of original images
    and phase images. It inherits from the torch.utils.data.Dataset class.

    Attributes:
    - X_orig (torch.Tensor): Tensor containing original images.
    - X_phase (torch.Tensor): Tensor containing phase images.

    Methods:
    - __getitem__(idx): Retrieve an item from the dataset by index.
    - __len__(): Get the total number of items in the dataset.

    """
    def __init__(self, X_orig, X_phase):
        """
        Initialize the ImageDataset.

        Parameters:
        - X_orig (torch.Tensor): Tensor containing original images.
        - X_phase (torch.Tensor): Tensor containing phase images.

        """
        self.X_orig = X_orig
        self.X_phase = X_phase
    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset by index.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        Tuple: A tuple containing the original image and the phase image.

        """
        return self.X_orig[idx], self.X_phase[idx]
    
    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        int: The total number of items in the dataset.

        """
        return len(self.X_orig) 

def run_autoencoder_phase_recovery():
    """
    Run the process of training an autoencoder for phase recovery.

    This function orchestrates the entire process of training an autoencoder model
    for phase recovery. It checks for the availability of CUDA device, loads the dataset,
    prepares data loaders, trains the model with different learning rates, evaluates
    the trained models, and saves exemplary images for visual comparison.

    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"# Using device: {device}")

    torch.cuda.empty_cache()

    X_orig_train, X_phase_train = get_dataset(0, 4500, IMG_SIZE)
    X_orig_test, X_phase_test = get_dataset(4500, 5000, IMG_SIZE)

    train_dataset = ImageDataset(X_orig_train, X_phase_train)
    val_dataset = ImageDataset(X_orig_test, X_phase_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    training_results = []
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
        training_results.append(train_model(lr, train_loader, val_loader))
    
    recovered_output = evaluate_models(training_results, val_loader)

    save_examplorary_images(X_orig_test, X_phase_test, recovered_output)
    
run_autoencoder_phase_recovery()