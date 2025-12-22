import torch
from dataset import MalariaDataset
import torchvision.transforms as transforms
import pandas as pd

def main():
    data = MalariaDataset(split='test', transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    # Load the trained model
    model = torch.load('models/best.pt', weights_only=False)
    model.eval()

    # Check if GPU is available and move model accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    filename, predictions = [], []
    with torch.no_grad():
        for idx in range(len(data)):
            img, _ = data[idx]
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            output = model(img)
            pred = torch.argmax(output, dim=1).item()
            filename.append(data.image_paths[idx].split('/')[-1])
            predictions.append(pred)
    # Save predictions to a CSV file
    df = pd.DataFrame({'filename': filename, 'prediction': predictions})
    df.to_csv('submission.csv', header=True, index=False)


if __name__ == "__main__":
    main()

