from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
from torchviz import make_dot
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from dataset import MalariaDataset
from auto_crop import AutoCrop



def visualize_model_prediction():
    model = torch.load('models/best.pt', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    torch.cuda.empty_cache()

    data = MalariaDataset(split='trainval', 
        transform=transforms.Compose([
        AutoCrop(threshold=5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))


    image, label = data[0]  # Get the first image and its label
    input = image.unsqueeze(0).to(device) # Add batch dimension
    labels = ['negative', 'positive']
    prediction = model(input).squeeze(0).softmax(0)
    predicted_label_idx = prediction.argmax().item()
    predicted_label = labels[predicted_label_idx]
    prediction_score = prediction[predicted_label_idx].item()
    print(prediction)

    print(f'Predicted: {predicted_label}, ({prediction_score:.2f}), True label: {label}')

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=predicted_label_idx, n_steps=200)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    # positive values point to pixels that increase the prediction score, negative attributions point to pixels that decrease the score
    # absolute attribution value show the magnitude of the effect regardless of direction

    # Add noise tunnel for smoother attributions
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=3, nt_type='smoothgrad_sq', target=predicted_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        cmap=default_cmap,
                                        show_colorbar=True)
if __name__ == "__main__":
    visualize_model_prediction()