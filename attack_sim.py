import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt

def run_adversarial_attack():
    # 1. Setup device and model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load your trained hackathon weights
    model.load_state_dict(torch.load("medical_resnet18_baseline.pth", map_location=device))
    model = model.to(device)
    model.eval() # Crucial: Attacks require evaluation mode

    # 2. Define image transformation pipeline (No normalization transformation here)
    transform_no_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Define preprocessing arrays for Foolbox (handles normalization automatically inside fmodel)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    # 3. Fetch a clean test sample
    base_data_path = "./dataset/chest_xray/chest_xray"
    test_dataset = ImageFolder(os.path.join(base_data_path, 'test'), transform=transform_no_norm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) # Shuffle to inspect random samples
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # 4. Initialize Foolbox Target passing the raw model object and our preprocessing dictionary
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    
    # Check original prediction
    initial_pred = torch.argmax(fmodel(images), dim=1).item()
    print(f"\n[SYSTEM READY] Patient Label: {'PNEUMONIA' if labels.item() == 1 else 'NORMAL'}")
    print(f"[AI DECISION] Original Prediction on Clean Image: {'PNEUMONIA' if initial_pred == 1 else 'NORMAL'}")

    # 5. Execute Projected Gradient Descent (PGD) Attack
    print("\n[ATTACK] Injecting stealthy adversarial noise map...")
    attack = fb.attacks.PGD()
    epsilons = [8/255] # Maximum pixel disturbance amplitude
    
    _, clipped_adversarial, success = attack(fmodel, images, labels, epsilons=epsilons)
    adv_image = clipped_adversarial[0]

    # 6. Check Adversarial Prediction
    adv_pred = torch.argmax(fmodel(adv_image), dim=1).item()
    print(f"[AI DECISION] Prediction on Adversarial Image: {'PNEUMONIA' if adv_pred == 1 else 'NORMAL'}")
    print(f"[ATTACK STATUS] Attack successful? {success.item()}")

    # 7. Generate Visual Proof for your Hackathon Presentation Slide
    print("\n[OUTPUT] Generating side-by-side diagnostic breakdown plots...")
    
    # Convert tensors back to viewable numpy arrays
    clean_np = images.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_np = adv_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    noise_np = np.abs(adv_np - clean_np)
    
    # Scale up noise visibility so judges can see the mathematical pattern
    if noise_np.max() > 0:
        noise_np = noise_np / noise_np.max()

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(clean_np)
    plt.title(f"Clean X-Ray\nAI Predicts: {'PNEUMONIA' if initial_pred == 1 else 'NORMAL'}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noise_np)
    plt.title("Injected Adversarial Noise\n(Amplified Visual Heatmap)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(adv_np)
    plt.title(f"Adversarial Image\nAI Predicts: {'PNEUMONIA' if adv_pred == 1 else 'NORMAL'}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("adversarial_proof_plot.png")
    print("Visual evidence saved successfully as 'adversarial_proof_plot.png'!")

if __name__ == "__main__":
    run_adversarial_attack()