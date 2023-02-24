from torchvision import transforms, models
import torch
import cv2


preprocess = lambda img: transforms.Compose([
    transforms.Resize(256),
    cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    test_folder = "K:\\Programming\\educational practice\\images\\test_data\\"
    train_folder = "K:\\Programming\\educational practice\\images\\train_data\\"

    img = cv2.imread("K:\\Programming\\educational practice\\images\\test_data\\day_images (1).jpg")

    img = preprocess(img)

    labels = ["day", "night"]

    resnet = models.resnet101()

    print(torch.cuda.is_available())

