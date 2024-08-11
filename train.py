import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

# Function to parse command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    
    # Required positional argument
    parser.add_argument('data_dir', type=str, help='Directory containing the training data')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (e.g., vgg16, resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

# Function to load and transform data
def load_data(data_dir):
    # Define data transforms for training and validation sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    validation_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=validation_transforms)
    
    # Define data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=16, num_workers=0, pin_memory=False)
    
    # Debugging: Print the number of images loaded
    print(f"Number of training images: {len(train_data)}")
    print(f"Number of validation images: {len(validation_data)}")

    return train_data, trainloader, validationloader

# Function to initialize the model
def initialize_model(arch, hidden_units):
    # Load a pre-trained model
    model = getattr(models, arch)(weights='IMAGENET1K_V1')
    
    # Freeze the parameters of the feature network
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the classifier or fully connected layer depending on the architecture
    if arch.startswith('vgg'):
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch.startswith('resnet'):
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError("Unsupported architecture")
    
    return model

# Function to train the model
def train_model(model, trainloader, validationloader, criterion, optimizer, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        running_loss = 0
        model.train()  # Set model to training mode
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            
            logps = model.forward(inputs)  # Forward pass
            loss = criterion(logps, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
        
        # Debugging: Print the running loss after each epoch
        print(f"Epoch {epoch+1} training loss: {running_loss/len(trainloader):.3f}")
        
        # After each epoch, validate the model
        validation_loss = 0
        accuracy = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                validation_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validationloader):.3f}")
        model.train()  # Reset model to training mode for next epoch

# Function to save the trained model checkpoint
def save_checkpoint(model, train_data, save_dir, arch, hidden_units):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

# Main function
def main():
    args = get_input_args()
    
    train_data, trainloader, validationloader = load_data(args.data_dir)
    
    model = initialize_model(args.arch, args.hidden_units)
    
    criterion = nn.NLLLoss()
    
    # Adjust optimizer based on the architecture
    if args.arch.startswith('vgg'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    elif args.arch.startswith('resnet'):
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    train_model(model, trainloader, validationloader, criterion, optimizer, args.epochs, args.gpu)
    
    save_checkpoint(model, train_data, args.save_dir, args.arch, args.hidden_units)

if __name__ == "__main__":
    main()
