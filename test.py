import torch
import numpy as np
from sklearn.metrics import classification_report

from models import mobilenet
from data_loader import get_test_dataloader

num_classes = 8
model = mobilenet(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('models/mob.pt', map_location=torch.device('cpu')))

test_loader, classes = get_test_dataloader(path='data/test', batch_size=16, IMAGE_SIZE = (300, 300))

def test(model, testloader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        y_pred = []
        y_actual = []
        model.eval()
        for _, (images,labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            y_actual += list(np.array(labels.detach().cpu()).flatten())
            _, predictes = torch.max(outputs, 1)
            y_pred += list(np.array(predictes.detach().cpu()).flatten())
            n_samples += labels.shape[0]

            n_correct += (predictes==labels).sum().item()
            
        y_actual = np.array(y_actual).flatten()
        y_pred = np.array(y_pred).flatten()
        
        acc = classification_report(y_actual, y_pred, target_names=classes)
        print(f"{acc}")


if __name__ == "__main__" :
        test(model, test_loader)