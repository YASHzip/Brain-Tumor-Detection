import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dl = DataLoader(
    datasets.ImageFolder('data/Training', tf),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 4)  # 4 classes
    ).to(device)

model = get_model()

model.load_state_dict(torch.load("BrainTumorRecognition.pth", map_location=device, weights_only=True))
model.eval()


loss_fn = nn.CrossEntropyLoss()


def main():
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x,y =x.to(device), y.to(device)

            logits = model(x)
            test_loss += loss_fn(logits, y).item() * y.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    test_loss /= len(test_dl.dataset)
    accuracy = 100.0 * correct / len(test_dl.dataset)

    print(f"Test Loss: {test_loss}, Test Accuracy: {accuracy}%")

if __name__ == "__main__":
    main()