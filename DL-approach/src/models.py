import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Dense, Flatten, ReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

import src.constants as C

#################################
### Custom 3DCNN architecture ###
#################################
class Custom3DCNN:
    def __init__(self, input_shape, depth = 4, initial_filters = 16, l2_strength = 0.01):
        self.input_shape = input_shape
        self.initial_filters = initial_filters
        self.l2_strength = l2_strength
        self.depth = depth

    def build_model(self):
        inputs = Input(shape = self.input_shape)

        # First convolutional layer
        x = Conv3D(filters = self.initial_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size = (2, 2, 2))(x)

        # Additional convolutional layers with increasing number of filters
        for i in range(self.depth):
            num_filters = self.initial_filters * (2 ** (i + 1))
            x = Conv3D(filters = num_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling3D(pool_size = (2, 2, 2))(x)

        # Last convolutional layer
        x = Conv3D(filters = self.initial_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = AveragePooling3D(pool_size = (2, 2, 2))(x)

        # Flattening the 3D outputs for the fully connected layer
        x = Flatten()(x)

        # Fully connected layer with L2 regularization
        x = Dense(units = 1, kernel_regularizer = l2(self.l2_strength))(x)

        # Create the model
        model = Model(inputs = inputs, outputs = x)
        return model

#################################
###===========================###
#################################

#########################
### SFCN architecture ###
#########################
    
class SFCN:
    """
    Simple Fully Convolutional Network (SFCN) for 3D data processing.
    """

    def __init__(self, input_shape, channels=[32, 64, 128, 256, 256, 64], output_dim=30, use_dropout=True, csize=C.CSIZE):
        self.input_shape = input_shape
        self.channels = channels
        self.output_dim = output_dim
        self.use_dropout = use_dropout
        self.csize = csize

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = inputs

        # Build the convolutional layers for the feature extractor
        for i, out_channel in enumerate(self.channels):
            maxpool = i < len(self.channels) - 1
            kernel_size = 3 if maxpool else 1
            padding = 'same' if maxpool else 'valid'

            # Add Conv3D layer
            x = Conv3D(out_channel, kernel_size, padding=padding, kernel_regularizer=l2(C.L2_STRENGTH))(x)
            # Add BatchNormalization layer
            x = BatchNormalization()(x)
            # Add ReLU activation layer
            x = ReLU()(x)
            # Add MaxPooling3D layer if maxpool is True
            if maxpool:
                x = MaxPooling3D(pool_size=2, strides=2)(x)

        # Build the classifier
        pool_size = int(self.csize / pow(2, len(self.channels) - 1))
        x = AveragePooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
        if self.use_dropout:
            x = Dropout(0.5)(x)
        x = Conv3D(self.output_dim, kernel_size=1, padding='valid')(x)
        x = Flatten()(x)

        # Create the model
        outputs = tf.nn.log_softmax(x, axis=1)
        model = Model(inputs=inputs, outputs=outputs)
        return model

#########################
###===================###
#########################


###########################
### ResNet architecture ###
###########################

class BasicBlock(tf.keras.layers.Layer):
    """
    BasicBlock is a building block of a ResNet model. It contains two 3D convolutional layers.
    Each convolutional layer is followed by batch normalization and ReLU activation.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Initializes the BasicBlock.

        :param in_planes: Number of input planes.
        :param planes: Number of output planes.
        :param stride: Stride size for the convolutional layer.
        :param downsample: Downsample function if needed for adjusting dimensions.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(planes, kernel_size=3, strides=stride,
                                            padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv3D(planes, kernel_size=3, strides=1,
                                            padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        """
        Forward pass for the block.

        :param x: Input tensor.
        :param training: Boolean, set to True if the model is in training mode.
        :return: Output tensor after passing through the block.
        """
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # Adjusting the identity if downsample is provided
        if self.downsample is not None:
            identity = self.downsample(x)

        # Adding the shortcut
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(tf.keras.layers.Layer):
    """
    Bottleneck is a more complex building block of a ResNet model, typically used in deeper networks.
    It has three convolutional layers: the first and the last are 1x1 convolutions, and the middle one is 3x3.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Initializes the Bottleneck block.

        :param in_planes: Number of input planes.
        :param planes: Number of intermediate planes.
        :param stride: Stride size for the second convolutional layer.
        :param downsample: Downsample function if needed for adjusting dimensions.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(planes, kernel_size=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv3D(planes, kernel_size=3, strides=stride,
                                            padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv3D(planes * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        """
        Forward pass for the block.

        :param x: Input tensor.
        :param training: Boolean, set to True if the model is in training mode.
        :return: Output tensor after passing through the block.
        """
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        # Third convolutional layer
        out = self.conv3(out)
        out = self.bn3(out, training=training)

        # Adjusting the identity if downsample is provided
        if self.downsample is not None:
            identity = self.downsample(x)

        # Adding the shortcut
        out += identity
        out = self.relu(out)

        return out

class ResNet(tf.keras.Model):
    """
    ResNet model, a popular neural network architecture for image classification.
    It consists of several layers of the BasicBlock or Bottleneck blocks.
    """
    def __init__(self, block, layers, num_classes=1000, channel_size=[64, 64, 128, 256, 512], dropout=True):
        """
        Initializes the ResNet model.

        :param block: Type of block to use (BasicBlock or Bottleneck).
        :param layers: List containing the number of blocks in each of the 4 layers of the model.
        :param num_classes: Number of output classes.
        :param channel_size: List of channel sizes for each layer.
        """
        super(ResNet, self).__init__()
        self.in_planes = channel_size[0]

        # Initial convolutional and pooling layers
        self.conv1 = tf.keras.layers.Conv3D(channel_size[0], kernel_size=7, strides=2,
                                            padding='same', use_bias=False,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = tf.keras.layers.BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling3D(pool_size=3, strides=2, padding='same')
        
        # Creating each ResNet layer
        self.layer1 = self._make_layer(block, channel_size[1], layers[0])
        self.layer2 = self._make_layer(block, channel_size[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channel_size[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channel_size[4], layers[3], stride=2)

        # Global average pooling and fully connected layer
        self.avgpool = tf.keras.layers.AvgPool3D(pool_size=(int(C.CSIZE / pow(2, len(channel_size))), 
                                                            int(C.CSIZE / pow(2, len(channel_size))), 
                                                            int(C.CSIZE / pow(2, len(channel_size)))), strides=1)
        # Dropout
        if dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)
        else:
            self.dropout = None
        # Flatten the output
        self.flatten = tf.keras.layers.Flatten()
        # Fully connected
        self.fc = tf.keras.layers.Dense(num_classes)

    def build_model(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = self.call(inputs)
        x = tf.nn.log_softmax(x, axis=1)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def call(self, x, training=False):
        """
        Forward pass for the model.

        :param x: Input tensor.
        :param training: Boolean, set to True if the model is in training mode.
        :return: Output tensor with class probabilities.
        """
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        # Passing through all the layers
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # Global average pooling and final dense layer
        x = self.avgpool(x)
        x = self.flatten(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Helper function to create a layer consisting of 'blocks' number of 'block' units.

        :param block: Block class (BasicBlock or Bottleneck).
        :param planes: Number of planes for this layer.
        :param blocks: Number of blocks in this layer.
        :param stride: Stride size for the first block of the layer.
        :return: A Sequential model consisting of the specified blocks.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv3D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

        layers = []
        # First block with potential downsampling
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return tf.keras.Sequential(layers)

def resnet18(input_shape, num_classes, channel_size, dropout):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, channel_size=channel_size, dropout=dropout)
    return model.build_model(input_shape=input_shape)
    
###########################
###=====================###
###########################