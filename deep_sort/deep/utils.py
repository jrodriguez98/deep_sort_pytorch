import torch
import torchvision

def calculate_mean_std_per_channel(train_dir, batch_size):
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=torchvision.transforms.ToTensor()),
        batch_size=1
    )

    mean_red, std_red = 0, 0
    mean_blue, std_blue = 0, 0
    mean_green, std_green = 0, 0
    dataset_lenght = len(trainloader.dataset)

    for input, labels in trainloader:

        mean_red += input[0][0].mean()
        mean_blue += input[0][1].mean()
        mean_green += input[0][2].mean()

        std_red += input[0][0].std()
        std_blue += input[0][1].std()
        std_green += input[0][2].std()

    mean_red, std_red = mean_red*batch_size/dataset_lenght, std_red*batch_size/dataset_lenght
    mean_blue, std_blue = mean_blue*batch_size/dataset_lenght, std_blue*batch_size/dataset_lenght
    mean_green, std_green = mean_green*batch_size/dataset_lenght, std_green*batch_size/dataset_lenght


    return [mean_red.numpy(), mean_blue.numpy(), mean_green.numpy()], [std_red.numpy(), std_blue.numpy(), std_green.numpy()]
