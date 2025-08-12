import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Define ViT Hyperparameters ---
# These parameters can be tuned for different datasets and model sizes
IMAGE_SIZE = 64  # We'll use 64x64 images for quicker demonstration
PATCH_SIZE = 8   # Size of the patches (e.g., 8x8)
NUM_CHANNELS = 3 # RGB images
NUM_CLASSES = 10 # Number of synthetic classes

# Derived parameters
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS # Flattened size of each patch

# Model specific parameters
EMBED_DIM = 768  # Dimension for patch embeddings (often 768 for ViT-Base)
NUM_HEADS = 12   # Number of attention heads in Multi-Head Self-Attention
MLP_RATIO = 4    # Ratio for MLP hidden layer size (e.g., EMBED_DIM * MLP_RATIO)
DROPOUT_RATE = 0.1 # Dropout rate for regularization
TRANSFORMER_LAYERS = 12 # Number of Transformer encoder blocks (often 12 for ViT-Base)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10 # Reduced epochs for quicker demonstration

# --- 2. Patch Embedding Layer ---
class PatchEmbedding(nn.Module):
    """
    PyTorch Module to split an image into patches, flatten them,
    and linearly project them to a higher dimension,
    then add positional embeddings.
    """
    def __init__(self, image_size, patch_size, num_channels, embed_dim, num_patches, dropout_rate):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Convolution to extract patches and project to embed_dim
        # This is equivalent to flattening patches and then a linear layer
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding (learnable)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # +1 for CLS token

        # Class Token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]

        # Project patches
        # Output shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        # Flatten patches: (batch_size, embed_dim, num_patches) -> (batch_size, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

# --- 3. Multi-Layer Perceptron (MLP) for Transformer Encoder ---
class MLP(nn.Module):
    """
    Helper module for the MLP block within the Transformer encoder.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU() # GELU is common in Transformers
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- 4. Transformer Encoder Block ---
class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block.
    Consists of Multi-Head Self-Attention and an MLP.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim,
                       out_features=embed_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        # Multi-head self-attention with skip connection and LayerNorm
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] # [0] for attention output
        # MLP with skip connection and LayerNorm
        x = x + self.mlp(self.norm2(x))
        return x

# --- 5. Build the ViT Model ---
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.
    """
    def __init__(self, image_size, patch_size, num_channels, num_classes,
                 embed_dim, num_heads, mlp_ratio, transformer_layers, dropout_rate):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, num_channels, embed_dim, num_patches, dropout_rate
        )

        self.transformer_encoder_blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
              for _ in range(transformer_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) # Classification head

    def forward(self, x):
        # Patch embedding and positional encoding
        x = self.patch_embedding(x)

        # Apply Transformer encoder blocks
        x = self.transformer_encoder_blocks(x)

        # Take the CLS token for classification (first token)
        x = self.norm(x[:, 0]) # LayerNorm and select CLS token

        # Classification head
        logits = self.head(x)
        return logits

# --- 6. Generate Synthetic Image Data ---
def generate_synthetic_image_data(num_samples=1000, img_size=(64, 64), num_channels=3, num_classes=10):
    """
    Generates synthetic image data and corresponding categorical labels.

    Args:
        num_samples (int): Number of synthetic samples to generate.
        img_size (tuple): Tuple (height, width) for image dimensions.
        num_channels (int): Number of color channels (e.g., 3 for RGB).
        num_classes (int): Number of distinct classes.

    Returns:
        tuple: (images, labels_encoded, label_encoder_instance)
               images (np.array): Synthetic image data (channels-first: N, C, H, W).
               labels_encoded (np.array): Integer-encoded labels.
               label_encoder_instance (LabelEncoder): The fitted LabelEncoder object.
    """
    print(f"Generating {num_samples} synthetic images of size {img_size}x{num_channels}...")
    # Generate random images (normalized to [0, 1])
    images = np.random.rand(num_samples, num_channels, img_size[0], img_size[1]).astype(np.float32)

    # Create categorical labels (e.g., 'class_0', 'class_1', etc.)
    class_names = [f'class_{i}' for i in range(num_classes)]
    labels_categorical = np.random.choice(class_names, num_samples)

    # Initialize and fit LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_categorical)

    print(f"Original categorical labels sample: {labels_categorical[:5]}")
    print(f"Encoded integer labels sample: {labels_encoded[:5]}")
    print(f"Classes found by LabelEncoder: {label_encoder.classes_}")

    return images, labels_encoded, label_encoder

# --- Main Execution ---
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate Data
    X_np, y_encoded_np, le = generate_synthetic_image_data(
        num_samples=2000, img_size=(IMAGE_SIZE, IMAGE_SIZE),
        num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES
    )

    # Split Data into Training and Testing Sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_encoded_np, test_size=0.2, random_state=42, stratify=y_encoded_np
    )

    # Convert NumPy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_np).to(device)
    y_train_tensor = torch.tensor(y_train_np).long().to(device) # Labels should be long() for CrossEntropyLoss
    X_test_tensor = torch.tensor(X_test_np).to(device)
    y_test_tensor = torch.tensor(y_test_np).long().to(device)

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nTraining data tensor shape: {X_train_tensor.shape}, Labels tensor shape: {y_train_tensor.shape}")
    print(f"Testing data tensor shape: {X_test_tensor.shape}, Labels tensor shape: {y_test_tensor.shape}")

    # Build the ViT Model
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        transformer_layers=TRANSFORMER_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # Print model summary (optional, but useful)
    print("\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- Train the Model ---
    print("\nTraining the Vision Transformer model...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Evaluate on Test Set (Validation) ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print("\nVision Transformer model training complete.")

    # --- Plot Training History ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Make Predictions (Example) ---
    print("\nMaking predictions on a few test samples:")
    model.eval() # Set to evaluation mode
    sample_indices = np.random.choice(len(X_test_np), 5, replace=False)
    sample_images_np = X_test_np[sample_indices]
    sample_true_labels_np = y_test_np[sample_indices]

    sample_images_tensor = torch.tensor(sample_images_np).to(device)

    with torch.no_grad():
        predictions_logits = model(sample_images_tensor)
        predictions_probs = torch.softmax(predictions_logits, dim=1)
        predicted_classes_encoded = torch.argmax(predictions_probs, dim=1).cpu().numpy()

    # Decode predictions back to original class names
    predicted_class_names = le.inverse_transform(predicted_classes_encoded)
    true_class_names = le.inverse_transform(sample_true_labels_np)

    for i in range(len(sample_images_np)):
        print(f"Sample {i+1}:")
        print(f"  True Label: {true_class_names[i]} (Encoded: {sample_true_labels_np[i]})")
        print(f"  Predicted Label: {predicted_class_names[i]} (Encoded: {predicted_classes_encoded[i]})")
        print(f"  Prediction Probabilities: {predictions_probs[i].cpu().numpy()}")
        print("-" * 30)

    # Visualize a few predictions
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(sample_images_np))):
        ax = plt.subplot(1, min(5, len(sample_images_np)), i + 1)
        # PyTorch images are C, H, W. Matplotlib expects H, W, C for RGB.
        img_display = np.transpose(sample_images_np[i], (1, 2, 0))
        plt.imshow(img_display)
        plt.title(f"True: {true_class_names[i]}\nPred: {predicted_class_names[i]}")
        plt.axis("off")
    plt.suptitle("Sample Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Save Model ---
    # os.makedirs("models", exist_ok=True)
    # model_filename = "models/vit_pytorch_model.pth" # PyTorch models typically saved as .pth or .pt
    # torch.save(model.state_dict(), model_filename)
    # print(f"✅ Saved PyTorch ViT model to {model_filename}")

    # --- Example of loading and using the saved model ---
    # print("\n--- Testing loading the saved model ---")
    # loaded_model = VisionTransformer(
    #     image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS,
    #     num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
    #     mlp_ratio=MLP_RATIO, transformer_layers=TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE
    # ).to(device)
    # loaded_model.load_state_dict(torch.load(model_filename))
    # loaded_model.eval() # Set to evaluation mode
    #
    # loaded_correct_val = 0
    # loaded_total_val = 0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = loaded_model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         loaded_total_val += labels.size(0)
    #         loaded_correct_val += (predicted == labels).sum().item()
    # loaded_accuracy = loaded_correct_val / loaded_total_val
    # print(f"Loaded Model Test Accuracy: {loaded_accuracy:.4f}")
    # assert abs(epoch_val_acc - loaded_accuracy) < 1e-6, "Loaded model predictions do not match original!"
    # print("Loaded model works correctly.")
