import torch
from torch import nn
from torchvision import models
import argparse
import json
from PIL import Image
import numpy as np

# Function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False
    
    if checkpoint['arch'].startswith('vgg'):
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )
    elif checkpoint['arch'].startswith('resnet'):
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Function to process the image
def process_image(image_path):
    img = Image.open(image_path)
    
    # Resize
    img = img.resize((256, 256))
    
    # Center crop
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    np_img = np.array(img)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    
    # Reorder dimensions
    np_img = np_img.transpose((2, 0, 1))
    
    return torch.from_numpy(np_img).float()

# Function to predict the class of an image
def predict(image_path, model, topk=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    img = process_image(image_path)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model.forward(img)
    
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
    
    return top_p.cpu().numpy()[0], top_classes

# Main function to parse arguments and run the prediction
def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained deep learning model")
    
    parser.add_argument('input', type=str, help="Path to the input image")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes")
    parser.add_argument('--category_names', type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available")
    
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.input, model, args.top_k, args.gpu)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]
    
    print("Predicted Classes and Probabilities:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()
