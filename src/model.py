import torch.nn as nn
import numpy as np

"""
The model has been taken from https://github.com/JanBrem/AI-NT-proBNP
"""


'''
计算在卷积操作中所需的填充（padding）大小，以便在给定步长（或降采样率）和卷积核大小的情况下，尽量使输出尺寸与输入尺寸接近或相同
步长为 1 ，输出尺寸与输入尺寸相同
当步长大于1时，即使使用了这个函数计算的填充值，输出尺寸仍然会小于输入尺寸
'''
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


'''
计算降采样率，并确保降采样是合理和有效的

降采样是将信号或数据集的采样率降低的过程
在时序数据、信号处理或图像处理中，降采样的主要目的是减少数据的数量，从而减少计算复杂性和存储需求，同时保留信号或数据集的主要特征。
在降采样之前应用适当的滤波器可以减少混叠效应，保留数据的重要特征。
'''
def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            "Number of samples for two consecutive blocks "
            "should always decrease by an integer factor."
        )
    return downsample


class ResBlock1d(nn.Module):
    """
    Residual network unit for unidimensional signals.
    """

    def __init__(
        self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
    ):
        if kernel_size % 2 == 0:
            raise ValueError("This implementation only supports odd `kernel_size`.")
        super(ResBlock1d, self).__init__()
        #super() 函数返回了 ResBlock1d 类的父类 即 nn.Module

        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False
        )        
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        '''
        当卷积层后面紧跟着BN层时，通常会选择不使用偏置
        因为批量归一化层本身会有一个偏移项，与偏置项的功能重叠
        '''

        '''
        例
        Conv1d 将一个长度为 100 的一维信号转换成另一个长度可能不同的一维信号
        而 Conv2d 将一个 28x28 的二维图像转换成另一个可能尺寸不同的二维图像
        '''
        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding,
            bias=False,
        )
        #在卷积神经网络中，增加步长是实现降采样的常见方法

        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        '''
        跳跃连接是一种常见的技巧，特别是在残差网络（ResNet）架构中。
        跳跃连接的主要思想是直接将某一层的输出添加到更深层的输出，从而跳过一到多个层。这种连接方式可以帮助避免深度网络中的梯度消失和梯度爆炸问题，同时也能增强模型的表达能力。
        例如，在一个简单的ResNet块中，输入首先通过一到两个卷积层进行处理，然后直接与块的输出相加，形成跳跃连接。

        skip_connection_layers 列表可能被用于存储那些需要在后续层与其他层的输出进行合并的层的输出。
        当网络向前传播时，我们可能会将某些层的输出保存到 skip_connection_layers 中。
        然后，在后续的层中，我们可以从 skip_connection_layers 中取出这些输出，并与当前层的输出合并，实现跳跃连接。
        '''
        # Skip connection
        skip_connection_layers = []

        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            '''
            确保输入特征的尺寸精确地减少一个因子 downsample
            窗口为 downsample: 每个池化窗口都会覆盖 downsample 个连续的输入值。然后，从这些值中选取最大值作为池化操作的输出。
            步长为 downsample: 每次移动池化窗口时，都会跳过 downsample 个输入值。这确保了每个池化窗口的输出都不重叠
            因此输出尺寸确实减少了一个因子 downsample。
            '''

            skip_connection_layers += [maxpool]

        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]

        # Build skip connection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """
        Residual unit.
        """

        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y

        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)

        # Sum skip connection and main connection
        x += y

        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """
    Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple should be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8
    ):
        super(ResNet1d, self).__init__()

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in,
            n_filters_out,
            kernel_size,
            bias=False,
            stride=downsample,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(
                n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
            )
            self.add_module("resblock1d_{0}".format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last  # * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """
        Implement ResNet1d forward propagation
        """

        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.mean(-1)

        logits = self.lin(x)
        return logits


def load_model():
    model = ResNet1d(
        input_dim=(3, 1000),
        blocks_dim=list(
            zip([32, 64, 128, 256, 384, 512], [1000, 500, 250, 125, 25, 5])
        ),
        n_classes=3,
        kernel_size=17,
        dropout_rate=0.0,
    )

    return model
