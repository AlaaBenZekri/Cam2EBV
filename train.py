import os
import models
import torch
from utils import free_memory
import utils
from dataset import Cam2BEVDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch.nn as nn
import torch.optim as optim

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_palette = utils.parse_convert_xml("convert_10.xml")
output_palette = utils.parse_convert_xml("convert_9+occl.xml")
input_shape = (256, 512)
output_shape = (256, 512)
train_label_dir = "data\\train\\bev+occlusion"
train_input_dir = ["data\\train\\front",
                   "data\\train\\left",
                   "data\\train\\rear",
                   "data\\train\\right"]
val_label_dir = "data\\val\\bev+occlusion"
val_input_dir = ["data\\val\\front",
                   "data\\val\\left",
                   "data\\val\\rear",
                   "data\\val\\right"]

				   
train_data = Cam2BEVDataset(train_input_dir, train_label_dir, input_shape, output_shape, input_palette, output_palette, device)
val_data = Cam2BEVDataset(val_input_dir, val_label_dir, input_shape, output_shape, input_palette, output_palette, device)

train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=5, shuffle=True, num_workers=0)

model = models.UNetXST((10, 256, 512), 1, 10, None, 5, 16, device)
model = model.to(device)

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
metric = MulticlassAccuracy(num_classes=10).to(device)

num_epochs = 5

torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in train_loader:

        labels = labels.max(dim=1)[1]
        images = [images[0]]
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_acc = metric(outputs.detach(), labels.detach()).item()
        running_acc += batch_acc
        
    train_loss = running_loss/len(train_loader)
    train_acc = running_acc/len(train_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
    
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:

            labels = labels.max(dim=1)[1]
            
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            batch_acc = metric(outputs.detach(), labels.detach()).item()
            running_acc += batch_acc    
    
    val_loss = running_loss/len(val_loader)
    val_acc = running_acc/len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
    print(epoch)
torch.autograd.set_detect_anomaly(False)