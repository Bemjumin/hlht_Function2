import torch
import torchvision
import torchvision.transforms as transforms
import datasets


def get_data10():
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.485, .456, .406],
                              std=[.229, .224, .225])])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False)
    return trainloader, testloader



def get_data100():
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.485, .456, .406],
                              std=[.229, .224, .225])])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False)
    #return trainloader, testloader
    return trainloader, testloader


def get_sst_data():
    sst2 = datasets.load_dataset(path='sst2', cache_dir='./data')
    trainloader = torch.utils.data.DataLoader(sst2["train"],batch_size=4)
    testloader = torch.utils.data.DataLoader(sst2["test"],batch_size=4)
    trainloader = zip(trainloader.dataset.data['sentence'], trainloader.dataset.data['label'])
    testloader = testloader.dataset.data['sentence']
    return trainloader, testloader



def get_data(dataset):
    if dataset == 'cifar10':
        return get_data10()
    elif dataset == 'cifar100':
        return get_data100()
    elif dataset == 'sst2':
        return get_sst_data()

