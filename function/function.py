# This is a sample Python script.
import torch
import torch.nn as nn


class FunctionEvaluation:
    def __init__(self, model, device, dataloader):

        """
        @description:
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            dataloader:可迭代的测试数据集
            }
        @return: None
        """
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.init_model(device)

    def init_model(self, device):
        """
        @description:将网络部署到环境设备上
        @param {
            device: 设备（GPU）
        }
        @return: None
        """
        self.model.eval().to(device)
        pass

    def prop_evaluate(self):
        """
        @description: 检测模型输出类型数量是否符合所用数据集类别数量
        @param: None
        @return: Bool
        """
        num = 0
        for name, layer in reversed(list(self.model.named_modules())):
            if isinstance(layer, nn.Linear):
                num = layer.bias.size(0)
                break
        if len(self.dataloader.dataset.classes) == num:
            print("The model is proper, number of classes is equal to the number of outputs: ", num)
            return True
        else:
            print("The model is not proper, number of classes is not equal to the number of outputs")
            print("The model has {} outputs, but the number of classes is {}".format(num, len(self.dataloader.dataset.classes)))
            return False

    def correct_evaluate(self, model_target):
        """
        @description: 检测输入模型是否与目标模型结构相同
        @param: model_target: 目标模型
        @return: None
        """
        for (name, layer), (name1, layer1) in zip(self.model.named_modules(), model_target.named_modules()):

            if name != name1:
                raise 'the model is not correct, please check the model'
            elif type(layer) == type(layer1):
                pass
            else:
                print('the layer of model {} should be {} but {} '.format(name, layer1, layer))

    def acc_evaluate(self):
        """
            @description: 检测模型在给定数据集下的准确率
            @param: None
            @return: float: 准确率（小于1）
        """
        if self.prop_evaluate():
            self.model.eval()
            total = 0
            correct = 0
            print('------------start test:-------------')
            with torch.no_grad():
                for data in self.dataloader:
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the dataset is: %d %%' % (
                    100 * correct / total))
            return correct / total
        else:
            raise print('the result does not match, please change check')
