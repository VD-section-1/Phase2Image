import os
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_dataset(start_id, end_id):
    """
    Load a dataset of grayscale and phase images.
    Parameters:
    - start_id (int): The starting index for the range of images to load.
    - end_id (int): The ending index for the range of images to load.

    This function loads a set of grayscale and phase images from specified directories. It iterates over a range of indices from start_id to end_id. 
    For each index, it constructs the file paths to the grayscale and phase images, opens the images, converts them to grayscale ('L'), and transforms them into tensors. 
    The tensors are then appended to the grayscale_images and phase_images lists.

    The function returns two lists: grayscale_images and phase_images. Each list contains the tensor representations of the images in the range from start_id to end_id.
    The images are expected to be in JPEG format and named in the format 'image_{i}.jpg', where {i} is the index of the image.
    """
    grayscale_dir = 'D:\github\Phase2Image\data\grayscale'
    phases_dir = 'D:\github\Phase2Image\data\phases'

    grayscale_images = []
    phase_images = []

    for i in range(start_id, end_id):
        grayscale_path = os.path.join(grayscale_dir, f'image_{i}.jpg')
        phase_path = os.path.join(phases_dir, f'image_{i}.jpg')

        grayscale_image = Image.open(grayscale_path).convert('L')
        phase_image = Image.open(phase_path).convert('L')

        grayscale_images.append(ToTensor()(grayscale_image))
        phase_images.append(ToTensor()(phase_image))

    return grayscale_images, phase_images


def split_dataset():
    """
    Split the dataset into training and testing sets.

    This function loads a dataset of 5000 grayscale and phase images using the load_dataset function. It then splits these images into training and testing sets. 
    The first 4500 images are used for training, and the remaining 500 images are used for testing.
    
    The function returns two tuples: the first tuple contains the training grayscale and phase images, and the second tuple contains the testing grayscale and phase images.
    """
    grayscale_images, phase_images = load_dataset(0, 5000)
    train_grayscale = grayscale_images[:4500]
    train_phase = phase_images[:4500]

    test_grayscale = grayscale_images[4500:]
    test_phase = phase_images[4500:]

    return (train_grayscale, train_phase), (test_grayscale, test_phase)



def train_model(lr, batch_size, epochs):
    """
    Train a DeepLabV3 model with a ResNet50 backbone for image recovery.

    Parameters:
    - lr (float): The learning rate for the Adam optimizer.
    - batch_size (int): The number of samples per batch.
    - epochs (int): The number of times the learning algorithm will work through the entire training dataset.

    This function first splits the dataset into training and testing sets using the split_dataset function. It then creates DataLoader instances for the training and testing sets.
    The training set is shuffled before each epoch, but the testing set is not.

    A DeepLabV3 model with a ResNet50 backbone is initialized. The first convolutional layer of the backbone is replaced to accept grayscale images. 
    The model is moved to the GPU if one is available. The model is trained for the specified number of epochs using the Mean Squared Error (MSE) loss and the Adam optimizer. 
    For each batch in the training set, the model's gradients are zeroed, the model's output is computed, the loss is calculated, and the model's parameters are updated.

    After each epoch, the model is switched to evaluation mode and the loss is computed for the testing set. However, the model's parameters are not updated during this phase.

    The function returns the trained model.
    """
    (train_grayscale, train_phase), (test_grayscale, test_phase) = split_dataset()
    train_loader = DataLoader(list(zip(train_grayscale, train_phase)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(test_grayscale, test_phase)), batch_size=batch_size, shuffle=False)

    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using {device} for training")

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        for i, (grayscale, phase) in enumerate(train_loader):
            grayscale = Variable(grayscale).to(device)
            phase = Variable(phase).to(device)

            optimizer.zero_grad()

            output = model(phase)['out']

            loss = loss_func(output, grayscale)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (grayscale, phase) in enumerate(test_loader):
                grayscale = Variable(grayscale).to(device)
                phase = Variable(phase).to(device)

                output = model(phase)['out']

                loss = loss_func(output, grayscale)

    return model

def evaluate_model(model, batch_size):
    """
    Evaluate a trained model on a testing dataset.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - batch_size (int): The number of samples per batch.

    This function evaluates a trained model on the test dataset. It first splits the dataset using the split_dataset function, but only uses the test set. 
    A DataLoader instance is created for the test set, without shuffling. The model is switched to evaluation mode and the Mean Squared Error (MSE) loss is initialized. 
    The function then iterates over the test set, computing the model's output and the loss for each batch. The losses for all batches are summed to compute the total loss.

    The function returns the average loss per batch in the test set. This is computed by dividing the total loss by the number of batches in the test set.
    """
    (_, _), (test_grayscale, test_phase) = split_dataset()
    test_loader = DataLoader(list(zip(test_grayscale, test_phase)), batch_size=batch_size, shuffle=False)
    loss_func = nn.MSELoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (grayscale, phase) in enumerate(test_loader):
            grayscale = Variable(grayscale).cuda()
            phase = Variable(phase).cuda()

            output = model(phase)['out']

            loss = loss_func(output, grayscale)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def display_images(model, batch_size):
    """
    Display original, phase, and reconstructed images for comparison.

    Parameters:
    - model (torch.nn.Module): The trained model to use for image reconstruction.
    - batch_size (int): The number of samples per batch.

    This function uses a trained model to reconstruct images from the test dataset. It first splits the dataset using the split_dataset function, but only uses the test set. 
    A DataLoader instance is created for the test set, without shuffling. The function then iterates over the test set, computing the model's output for each batch and appending 
    the output to the reconstructed_images list.

    After all images have been reconstructed, the function displays the first 20 original images, phase images, and reconstructed images side by side for comparison.     
    """
    (_, _), (test_grayscale, test_phase) = split_dataset()

    test_loader = DataLoader(list(zip(test_grayscale, test_phase)), batch_size=batch_size, shuffle=False)

    reconstructed_images = []
    model.eval()
    with torch.no_grad():
        for i, (grayscale, phase) in enumerate(test_loader):
            grayscale = Variable(grayscale).cuda()
            phase = Variable(phase).cuda()

            output = model(phase)['out']
            reconstructed_images.append(output.squeeze(0).cpu())

    for i in range(20):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(test_grayscale[i].squeeze(0), cmap='gray')
        ax[0].set_title('Original Image')

        ax[1].imshow(test_phase[i].squeeze(0), cmap='gray')
        ax[1].set_title('Phase Image')

        ax[2].imshow(reconstructed_images[i][0][0], cmap='gray')
        ax[2].set_title('Improved Image')
        plt.show()


def DeepLabV3Plus():
    """
    Train and evaluate a DeepLabV3 model with a ResNet50 backbone for image recovery.

    This function trains a DeepLabV3 model with a ResNet50 backbone for image recovery using the train_model function. 
    It then evaluates the model using the evaluate_model function and displays the original, phase, and reconstructed images using the display_images function.
    """

    lr = 0.005
    batch_size = 16
    epochs = 20

    print("Training the model...")
    model = train_model(lr, batch_size, epochs)

    print("Evaluating the model...")
    avg_loss = evaluate_model(model, batch_size)
    print(f"Average loss: {avg_loss}")

    print("Displaying images...")
    display_images(model, batch_size)

DeepLabV3Plus()