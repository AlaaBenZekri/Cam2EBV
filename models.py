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
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

utils.make_reproduceable()
#device =  torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = models.UNetXST((10, 256, 512), 4, 10, None, 4, 16, device)
weights = os.listdir("./model_checkpoints")[-1]
model.load_state_dict(torch.load("./model_checkpoints/"+weights))
model = model.to(device)

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
metric = MulticlassAccuracy(num_classes=10).to(device)
save_samples_train = 50
save_samples_val = 350
save_weights = 500
num_epochs = 5

saved_train_samples = []
saved_val_samples = []
#torch.autograd.set_detect_anomaly(True)

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(1, num_epochs):
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    total_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for filenames, images, oh_labels in train_loader:

        labels = oh_labels.max(dim=1)[1]
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_samples += labels.size(0)
        total_batches += 1
        running_loss += loss.item()
        batch_acc = metric(outputs.detach(), labels.detach()).item()
        running_acc += batch_acc
        progress_bar.set_postfix(loss = running_loss/total_samples, accuracy = running_acc/total_batches)
        progress_bar.update()
        train_loss_list.append(running_loss/total_samples)
        train_acc_list.append(running_acc/total_batches)
        if total_batches%save_samples_train==0:
            os.mkdir("./data/train_predictions/train_prediction_epoch_"+str(epoch)+"_step_"+str(total_batches))
            utils.save_samples(filenames, oh_labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), output_palette, path="./data/train_predictions/train_prediction_epoch_"+str(epoch)+"_step_"+str(total_batches))
        if total_batches%save_weights==0:
            torch.save(model.state_dict(), "./model_checkpoints/weights_epoch_"+str(epoch)+"_step_"+str(total_batches)+".pth")
    train_loss = running_loss/len(train_loader)
    train_acc = running_acc/len(train_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_acc:.4f}")
    
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    total_batches = 0
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    with torch.no_grad():
        for filenames, images, oh_labels in val_loader:

            labels = oh_labels.max(dim=1)[1]
            
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_samples += labels.size(0)
            total_batches += 1
            running_loss += loss.item()
            batch_acc = metric(outputs.detach(), labels.detach()).item()
            running_acc += batch_acc    
            total_batches += 1
            progress_bar.set_postfix(loss = running_loss/total_samples, accuracy = running_acc/total_batches)
            progress_bar.update()
            val_loss_list.append(running_loss/total_samples)
            val_acc_list.append(running_acc/total_batches)
            
            if total_batches%save_samples_val==0:
                os.mkdir("./data/val_predictions/val_prediction_epoch_"+str(epoch))
                utils.save_samples(filenames, oh_labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), output_palette, path="./data/val_predictions/val_prediction_epoch_"+str(epoch))

    val_loss = running_loss/len(val_loader)
    val_acc = running_acc/len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
    print(epoch)
#torch.autograd.set_detect_anomaly(False)
