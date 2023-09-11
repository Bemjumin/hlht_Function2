# This is a sample Python script.
import torch
import argparse
from performance.performance import PerformEvaluation
from pytorchcv.model_provider import get_model as ptcv_get_model
from function.function import FunctionEvaluation
import torch.nn as nn
from dataset import get_data
from select_imagenetmodel import load_imagenet_model
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--root', default='model_weights/vgg16_cifar10.pth', type=str, metavar='PATH',
                        help='path to the pretrain model')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pattern', type=str, default='test')

    # opt = vars(parser.parse_args())     # Object: Namespace -> 字典
    arguments = parser.parse_args()
    return arguments


def test(test_dataloader, device, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def is_save_best(correct, total, best_acc):
    if (correct / total) > best_acc:
        best_acc = (correct / total)
        is_best = True
    else:
        is_best = False
    return best_acc, is_best


def train(train_dataloader, device, model, args):

    total = 0
    correct = 0
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    model = model.eval()
    model = model.to(device)
    #optimizer = torch.optim.SGD(model.output.fc3.parameters(), lr=0.001, momentum=0.9)   #vgg16
    optimizer = torch.optim.SGD(model.output.parameters(), lr=0.001, momentum=0.9)    #resnet50
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # 输入数据
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 统计损失
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 打印日志
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                print('Accuracy of the network on the train images: %d %%' % (
                100 * correct / total))
                best_acc, is_save = is_save_best(correct, total, best_acc)
                if is_save:
                    torch.save(model.state_dict(), args.root)
                    print('save model')
                    print('Accuracy of the network on the 10000 test images: %d %%' % (
                            100 * correct / total))
            if best_acc > 0.5:
                torch.save(model.state_dict(), args.root)
                break


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载模型
    model = load_imagenet_model(args.model, args.dataset, args.root, device)
    train_dataloader, test_dataloader = get_data(args.dataset)
    if args.pattern == 'test':
        print('----------功能测试开始----------')
        f = FunctionEvaluation(model, device, test_dataloader)
        f.acc_evaluate()
        model_wrong = ptcv_get_model("vgg16", pretrained=False)
        f = FunctionEvaluation(model_wrong, device, test_dataloader)
        f.prop_evaluate()
        print('----------功能测试结束----------')

        print('----------性能测试开始----------')
        p = PerformEvaluation(model, device, test_dataloader)
        p.computation()
        p.occupancy_rate()
        p.compute_sparse_degree()
        print('----------性能测试结束----------')
    elif args.pattern == 'train':
        train(train_dataloader, device, model, args)



# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    args = collect_args()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
