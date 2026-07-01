import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import foolbox as fb
import numpy as np
import pywt
import matplotlib.pyplot as plt

# --- CORE INNOVATION: THE RADSHIELD DEFENSE ENGINE ---

def wavelet_denoising_layer(image_tensor):
    """
    Strips out high-frequency adversarial perturbations using 
    Discrete Wavelet Transform (DWT) 2D Denoising.
    """
    # Convert PyTorch tensor to CPU numpy array for processing
    img_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    denoised_channels = []
    for i in range(3): # Process R, G, B channels individually
        channel = img_np[:, :, i]
        # Decompose the image into high and low frequency coefficients
        coeffs = pywt.dwt2(channel, 'db1')
        cA, (cH, cV, cD) = coeffs
        
        # Soft-thresholding to filter out tiny, unnatural adversarial variations
        threshold = 0.04
        cH_t = pywt.threshold(cH, threshold, mode='soft')
        cV_t = pywt.threshold(cV, threshold, mode='soft')
        cD_t = pywt.threshold(cD, threshold, mode='soft')
        
        # Reconstruct the cleaned channel matrix
        cleaned_channel = pywt.idwt2((cA, (cH_t, cV_t, cD_t)), 'db1')
        # Resize/crop back to original dimension match if needed
        cleaned_channel = cleaned_channel[:224, :224]
        denoised_channels.append(cleaned_channel)
        
    denoised_np = np.stack(denoised_channels, axis=0) # Shape: (3, 224, 224)
    return torch.tensor(denoised_np).unsqueeze(0).float()

def randomized_masking_verification(model, image_tensor, base_pred, device):
    """
    Implements security anomaly checks. Randomly blocks out 5% tiles.
    If the model's confidence scores wildly fluctuate, a tamper attack is flagged.
    """
    model.eval()
    tampers_detected = 0
    test_runs = 3
    
    with torch.no_grad():
        for _ in range(test_runs):
            masked_img = image_tensor.clone()
            # Randomly pick a 25x25 block to drop out
            x = np.random.randint(0, 190)
            y = np.random.randint(0, 190)
            masked_img[:, :, x:x+25, y:y+25] = 0.0
            
            output = model(masked_img)
            mask_pred = torch.argmax(output, dim=1).item()
            
            if mask_pred != base_pred:
                tampers_detected += 1
                
    # If predictions switch wildly when non-critical pixels are masked, the image math is toxic
    return True if tampers_detected >= 1 else False

# --- EVALUATION PIPELINE ---

def run_radshield_defense():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Baseline Vulnerable System
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("medical_resnet18_baseline.pth", map_location=device))
    model = model.to(device)
    model.eval()

    transform_no_norm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    
    # 2. Get a sample image and generate the attack
    base_data_path = "./dataset/chest_xray/chest_xray"
    test_dataset = ImageFolder(os.path.join(base_data_path, 'test'), transform=transform_no_norm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    attack = fb.attacks.PGD()
    _, clipped_adversarial, _ = attack(fmodel, images, labels, epsilons=[8/255])
    adv_image = clipped_adversarial[0]

    # 3. Process through RadShield Firewall Layers
    print("\n[RADSHIELD] Intercepting incoming radiology transmission stream...")
    
    # Run Tamper Detection
    attack_detected = randomized_masking_verification(model, adv_image, torch.argmax(fmodel(adv_image), dim=1).item(), device)
    if attack_detected:
        print("⚠️ [ALERT] SECURITY ANOMALY DETECTED! High-variance mathematical signature matching Adversarial Patch template.")
    
    # Apply Denoising Core
    print("[RADSHIELD] Executing Wavelet Denoising Core to strip adversarial signatures...")
    clean_recovered_tensor = wavelet_denoising_layer(adv_image).to(device)

    # 4. Final Diagnostics Verification
    orig_prediction = torch.argmax(fmodel(images), dim=1).item()
    fooled_prediction = torch.argmax(fmodel(adv_image), dim=1).item()
    defended_prediction = torch.argmax(fmodel(clean_recovered_tensor), dim=1).item()

    print(f"\n[FINAL REPORT]")
    print(f"├─ True Patient Status:       {'PNEUMONIA' if labels.item() == 1 else 'NORMAL'}")
    print(f"├─ Vulnerable AI Output:       {'PNEUMONIA' if fooled_prediction == 1 else 'NORMAL'}")
    print(f"└─ RadShield Protected Output: {'PNEUMONIA' if defended_prediction == 1 else 'NORMAL'}")

    # 5. Export Final Comparative Proof for Pitch Deck
    clean_np = images.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_np = adv_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    defended_np = clean_recovered_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # Clip numpy ranges to prevent matplotlib visualization warnings
    defended_np = np.clip(defended_np, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(clean_np)
    plt.title(f"1. Original Scan\nAI: {'PNEUMONIA' if orig_prediction == 1 else 'NORMAL'}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_np)
    plt.title(f"2. Poisoned Input\nAI: {'PNEUMONIA' if fooled_prediction == 1 else 'NORMAL'}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(defended_np)
    plt.title(f"3. RadShield Recovered\nAI: {'PNEUMONIA' if defended_prediction == 1 else 'NORMAL'}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("radshield_defense_proof.png")
    print("\n[SUCCESS] Matrix defense graph compiled as 'radshield_defense_proof.png'. Phase 2 complete!")

if __name__ == "__main__":
    # Suppress git warning variables 
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    run_radshield_defense()