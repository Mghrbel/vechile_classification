import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adamax
from sklearn.metrics import accuracy_score

from models import vgg
from data_loader import get_train_val_dataloader
from data_augmenter import create_augmented_data


def train(model, train_loader, criterion, optimizer, val_loader, epochs=25):
    train_losses, val_losses, train_acc, val_accuracies, train_acc_epoch, val_acc_epoch, y_actual, y_pred = [], [], [], [], [], [], [], []
    wait = 0
    patience = 5
    best_acc = 0.0
    min_loss = np.Inf
    if not os.path.exists("models") : os.mkdir("models")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} : ")
        y_actual, y_pred = [], []
        train_loss, val_loss = 0, 0

        # Train the model
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Loss and accuracy
            train_loss += loss.item()
            
            _, predictes = torch.max(outputs, dim=1)
            y_actual += list(labels.data.cpu().numpy().flatten()) 
            y_pred += list(predictes.detach().cpu().numpy().flatten())

            acc = accuracy_score(y_actual, y_pred)
            end_time = time.time()
            ETA = (end_time-start_time)*(n_steps-step)
            print('\rstep [%d/%d] : loss = %.4f, accuracy = %.4f, ETA = %d sec ' % (step, n_steps, train_loss/(step+1), acc, ETA), end='')
        train_acc.append(accuracy_score(y_actual, y_pred))
        print('\rstep [%d/%d] : loss = %.4f, accuracy = %.4f ' % (step+1, n_steps, train_loss/(step+1), acc), end='')


        # Evaluate the model
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Loss and accuracy
            val_loss += loss.item()
            _,predictes=torch.max(outputs, 1)
            y_actual += list(labels.data.cpu().numpy().flatten()) 
            y_pred += list(predictes.detach().cpu().numpy().flatten())
        
        val_accuracies.append(accuracy_score(y_actual, y_pred))

        # Average losses and accuracies
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        training_acc = train_acc[-1]
        val_acc = val_accuracies[-1]
        train_acc_epoch.append(training_acc)
        val_acc_epoch.append(val_acc)

        # Updating best validation accuracy
        if best_acc < val_acc:
            best_acc = val_acc

        print(", val_loss = %.4f, val_accuracy = %.4f" % (val_loss, val_acc))

        wait += 1
        # Saving best model
        if min_loss >= val_loss:
            torch.save(model.state_dict(), 'models/vgg.pt')
            print("val_loss improved from %.4f to %.4f, model saved to models/vgg.pt" % (min_loss, val_loss))
            min_loss = val_loss
            wait = 0

        else :
            print("val_loss did not improve, skip checkpoint")
        print('=' * 80)

        # Early stopping
        if wait >= patience:
            print(f"val_loss did not improve for series of {wait} epochs, Apply early stopping")
            break
        
    return train_losses, val_losses, train_acc, val_acc, train_acc_epoch, val_acc_epoch


if __name__ == "__main__" :
        create_augmented_data()

        num_classes = 8
        batch_size = 16
        IMAGE_SIZE = (300, 300)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        epochs = 100
        learning_rate = 0.0008

        model = vgg(num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adamax(model.parameters(), lr=learning_rate)


        train_loader, val_loader = get_train_val_dataloader('data/train/augmented_data', batch_size, IMAGE_SIZE)
        n_steps = len(train_loader)
        print("\nStart training :")
        train_losses, val_losses, train_acc, val_acc, train_acc_epoch, val_acc_epoch = train(model, train_loader, criterion, optimizer, val_loader, epochs)