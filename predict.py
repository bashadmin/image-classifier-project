import torch
from torchvision import models, transforms
from PIL import Image
import json
import argparse

# Function to parse command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    
    # Required positional arguments
    parser.add_argument('input', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

# Function to load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process the input image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze_(0)

# Function to make predictions
def predict(image, model, top_k, device):
    model.eval()
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_ps, top_classes = ps.topk(top_k, dim=1)
    return top_ps.cpu().numpy()[0], top_classes.cpu().numpy()[0]

# Function to map the classes to names
def map_classes(classes, cat_to_name):
    if cat_to_name:
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        return [cat_to_name[str(class_idx)] for class_idx in classes]
    return classes

# Main function
def main():
    args = get_input_args()
    
    model = load_checkpoint(args.checkpoint)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    image = process_image(args.input)
    
    probs, classes = predict(image, model, args.top_k, device)
    
    class_names = map_classes(classes, args.category_names)
    
    for prob, class_name in zip(probs, class_names):
        print(f"Class: {class_name}, Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
