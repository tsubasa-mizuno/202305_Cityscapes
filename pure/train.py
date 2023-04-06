import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from pure import CityscapesDataset
from pure import SegmentationModel

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = CityscapesDataset('path/to/cityscapes', split='train', transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Define model
model = SegmentationModel(num_classes=19)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        # Get inputs and labels
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if i % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item():.4f}')
