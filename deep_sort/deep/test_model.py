import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os
from loader import CustomFolder
from model_sq import Net


# compute features
def process_loader(loader, net, device):
    all_features = torch.tensor([]).float()
    all_labels = torch.tensor([]).long()

    for idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        all_features = torch.cat((all_features, features), dim=0)
        # print(labels)
        all_labels = torch.cat((all_labels, labels))

    return all_features, all_labels


def get_model_name(model_path):
    model_name = model_path.split(sep="/")[-1]
    model_name = model_name.split(sep=".")[0]

    return model_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='data', type=str)
    parser.add_argument("--model", default='checkpoint/ckpt.t7', type=str)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", default=0, type=int)
    args = parser.parse_args()

    # device
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True

    # data loader
    root = args.data_dir
    model_path = args.model
    query_dir = os.path.join(root, "query")
    gallery_dir = os.path.join(root, "gallery")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    queryloader = torch.utils.data.DataLoader(
        CustomFolder(query_dir, transform=transform),
        batch_size=64, shuffle=False
    )
    galleryloader = torch.utils.data.DataLoader(
        CustomFolder(gallery_dir, transform=transform),
        batch_size=64, shuffle=False
    )

    # net definition
    net = Net(num_classes=34, reid=True)
    assert os.path.isfile(model_path), "Error: no checkpoint file found!"
    print(f'Loading from {model_path}')
    checkpoint = torch.load(model_path)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict, strict=False)
    net.eval()
    net.to(device)

    with torch.no_grad():
        query_features, query_labels = process_loader(queryloader)
        gallery_features, gallery_labels = process_loader(galleryloader)

    # gallery_labels -= 2

    # save features
    features = {
        "qf": query_features,
        "ql": query_labels,
        "gf": gallery_features,
        "gl": gallery_labels
    }

    if not os.path.exists("features"):
        os.makedirs("features")

    model_name = get_model_name(model_path)

    print(f"Guardando features en features/{model_name}.pth")
    torch.save(features, f"features/{model_name}.pth")
