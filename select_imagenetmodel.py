import torch
#from norm_layer import get_normalize_layer
from pytorchcv.model_provider import get_model as ptcv_get_model
import os


def load_imagenet_model(model_name, dataset, root, device):
    #normalize_layer = get_normalize_layer('imagenet')
    model = None
    if model_name == 'vgg16':
        if dataset == 'cifar10':
            model = ptcv_get_model("vgg16", pretrained=False)
            model.output.fc3 = torch.nn.Linear(4096, 10)
            model.num_classes = 10
            if os.path.exists(root):
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
            else:
                torch.save(model.state_dict(), root)
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
        elif dataset == 'cifar100':
            model = ptcv_get_model("vgg16", pretrained=False)
            model.output.fc3 = torch.nn.Linear(4096, 100)
            model.num_classes = 100
            if os.path.exists(root):
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
            else:
                torch.save(model.state_dict(), root)
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
    elif model_name == 'resnet50':
        if dataset == 'cifar10':
            model = ptcv_get_model("resnet50", pretrained=False)
            model.output = torch.nn.Linear(2048, 10)
            model.num_classes = 10

            if os.path.exists(root):
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
            else:
                torch.save(model.state_dict(), root)
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
        if dataset == 'cifar100':
            model = ptcv_get_model("resnet50", pretrained=False)
            model.output = torch.nn.Linear(2048, 100)
            model.num_classes = 100

            if os.path.exists(root):
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
            else:
                torch.save(model.state_dict(), root)
                checkpoint = torch.load(root, map_location=device)
                model.load_state_dict(checkpoint)
    #model.load_state_dict(checkpoint)
    return model
    #return torch.nn.Sequential(normalize_layer, model)
