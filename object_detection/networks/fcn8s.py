
from torch import nn
from .base import BaseNetwork
from object_detection.utils.misc import get_upsampling_weight


# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s1(BaseNetwork):
    """
    FCN 8s Network
    """
    def __init__(self, *args, **kwargs):
        super(FCN8s1, self).__init__(*args, **kwargs)

    def make_network(self):
        """
        Make Network
        :return:
        """
        n_class = self.n_classes
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.bn6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096,kernel_size= 1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)
        self.score_fr_bn = nn.BatchNorm2d(n_class)
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=n_class,kernel_size= 1)
        self.score_pool4_bn = nn.BatchNorm2d(n_class)
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)
        self.score_pool3_bn = nn.BatchNorm2d(n_class)


        self.upscore2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4, stride=2, bias=False)
        self.upscore2_bn = nn.BatchNorm2d(n_class)
        self.upscore8 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=16, stride=8, bias=False)
        self.upscore8_bn = nn.BatchNorm2d(n_class)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4, stride=2, bias=False)

        self._initialize_weights()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)  # 1/2

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)  # 1/4

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.pool5(h)  # 1/32

        h = self.relu6(self.bn6(self.fc6(h)))
        h = self.drop6(h)

        h = self.relu7(self.bn7(self.fc7(h)))
        h = self.drop7(h)

        h = self.score_fr_bn(self.score_fr(h))
        h = self.upscore2_bn(self.upscore2(h))
        upscore2 = h  # 1/16

        h = self.score_pool4_bn(self.score_pool4(pool4))
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3_bn(self.score_pool3(pool3))
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8_bn(self.upscore8(h))
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.zero_()

# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s2(BaseNetwork):
    """
    FCN 8s Network
    """
    def __init__(self, *args, **kwargs):
        super(FCN8s2, self).__init__(*args, **kwargs)

    def make_network(self):
        """
        Make Network
        :return:
        """
        n_class = 1
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.bn6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096,kernel_size= 1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)
        self.score_fr_bn = nn.BatchNorm2d(n_class)
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=n_class,kernel_size= 1)
        self.score_pool4_bn = nn.BatchNorm2d(n_class)
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)
        self.score_pool3_bn = nn.BatchNorm2d(n_class)


        self.upscore2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4, stride=2, bias=False)
        self.upscore2_bn = nn.BatchNorm2d(n_class)
        self.upscore8 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=16, stride=8, bias=False)
        self.upscore8_bn = nn.BatchNorm2d(n_class)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4, stride=2, bias=False)

        self._initialize_weights()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)  # 1/2

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)  # 1/4

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.pool5(h)  # 1/32

        h = self.relu6(self.bn6(self.fc6(h)))
        h = self.drop6(h)

        h = self.relu7(self.bn7(self.fc7(h)))
        h = self.drop7(h)

        h = self.score_fr_bn(self.score_fr(h))
        h = self.upscore2_bn(self.upscore2(h))
        upscore2 = h  # 1/16

        h = self.score_pool4_bn(self.score_pool4(pool4))
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3_bn(self.score_pool3(pool3))
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8_bn(self.upscore8(h))
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.zero_()



# class FCN8sAtOnce(FCN8s):
#
#     def forward(self, x):
#         h = x
#         h = self.relu1_1(self.conv1_1(h))
#         h = self.relu1_2(self.conv1_2(h))
#         h = self.pool1(h)
#
#         h = self.relu2_1(self.conv2_1(h))
#         h = self.relu2_2(self.conv2_2(h))
#         h = self.pool2(h)
#
#         h = self.relu3_1(self.conv3_1(h))
#         h = self.relu3_2(self.conv3_2(h))
#         h = self.relu3_3(self.conv3_3(h))
#         h = self.pool3(h)
#         pool3 = h  # 1/8
#
#         h = self.relu4_1(self.conv4_1(h))
#         h = self.relu4_2(self.conv4_2(h))
#         h = self.relu4_3(self.conv4_3(h))
#         h = self.pool4(h)
#         pool4 = h  # 1/16
#
#         h = self.relu5_1(self.conv5_1(h))
#         h = self.relu5_2(self.conv5_2(h))
#         h = self.relu5_3(self.conv5_3(h))
#         h = self.pool5(h)
#
#         h = self.relu6(self.fc6(h))
#         h = self.drop6(h)
#
#         h = self.relu7(self.fc7(h))
#         h = self.drop7(h)
#
#         h = self.score_fr(h)
#         h = self.upscore2(h)
#         upscore2 = h  # 1/16
#
#         h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
#         h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
#         score_pool4c = h  # 1/16
#
#         h = upscore2 + score_pool4c  # 1/16
#         h = self.upscore_pool4(h)
#         upscore_pool4 = h  # 1/8
#
#         h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
#         h = h[:, :,
#               9:9 + upscore_pool4.size()[2],
#               9:9 + upscore_pool4.size()[3]]
#         score_pool3c = h  # 1/8
#
#         h = upscore_pool4 + score_pool3c  # 1/8
#
#         h = self.upscore8(h)
#         h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
#
#         return h
#
#     def copy_params_from_vgg16(self, vgg16):
#         features = [
#             self.conv1_1, self.relu1_1,
#             self.conv1_2, self.relu1_2,
#             self.pool1,
#             self.conv2_1, self.relu2_1,
#             self.conv2_2, self.relu2_2,
#             self.pool2,
#             self.conv3_1, self.relu3_1,
#             self.conv3_2, self.relu3_2,
#             self.conv3_3, self.relu3_3,
#             self.pool3,
#             self.conv4_1, self.relu4_1,
#             self.conv4_2, self.relu4_2,
#             self.conv4_3, self.relu4_3,
#             self.pool4,
#             self.conv5_1, self.relu5_1,
#             self.conv5_2, self.relu5_2,
#             self.conv5_3, self.relu5_3,
#             self.pool5,
#         ]
#         for l1, l2 in zip(vgg16.features, features):
#             if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
#                 assert l1.weight.size() == l2.weight.size()
#                 assert l1.bias.size() == l2.bias.size()
#                 l2.weight.data.copy_(l1.weight.data)
#                 l2.bias.data.copy_(l1.bias.data)
#         for i, name in zip([0, 3], ['fc6', 'fc7']):
#             l1 = vgg16.classifier[i]
#             l2 = getattr(self, name)
#             l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
#             l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
