import numpy as np

def count_flops_dense(in_features, out_features, bias=True, activation=True):
  flops = (2 * in_features - 1) * out_features
  if bias:
    flops += out_features
  if activation:
    flops += out_features
  # print(flops)
  return flops


def count_flops_conv(height,
                     width,
                     in_channels,
                     out_channels,
                     kernel_size,
                     stride=1,
                     padding=0,
                     bias=True,
                     activation=True):
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * 2
  n = kernel_size[0] * kernel_size[1] * in_channels
  flops_per_instance = 2 * n - 1
  out_height = (height - kernel_size[0] + 2 * padding) / stride + 1
  out_width = (width - kernel_size[1] + 2 * padding) / stride + 1
  num_instances_per_channel = out_height * out_width
  flops_per_channel = num_instances_per_channel * flops_per_instance
  total_flops = out_channels * flops_per_channel
  if bias:
    total_flops += out_channels * num_instances_per_channel
  if activation:
    total_flops += out_channels * num_instances_per_channel

  # print(height, width, in_channels, out_channels, total_flops)
  return total_flops


def count_flops_max_pool(height,
                         width,
                         channels,
                         kernel_size,
                         stride=None,
                         padding=0):
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * 2
  stride = kernel_size if stride is None else stride
  if isinstance(stride, int):
    stride = [stride] * 2
  flops_per_instance = kernel_size[0] * kernel_size[1]
  out_height = (height - kernel_size[0] + 2 * padding) / stride[0] + 1
  out_width = (width - kernel_size[1] + 2 * padding) / stride[1] + 1
  num_instances_per_channel = out_height * out_width
  flops_per_channel = num_instances_per_channel * flops_per_instance
  total_flops = channels * flops_per_channel
  return total_flops


def count_flops_avg_pool(height, width, channels):
  return channels * height * width


def vgg16_bn_flops(num_units, n_classes):
  flops = count_flops_conv(32, 32, num_units[0], num_units[1], 3, padding=1) + \
          count_flops_conv(32, 32, num_units[1], num_units[2], 3, padding=1) # + \
          # count_flops_max_pool(32, 32, num_units[2], 2)

  flops += count_flops_conv(16, 16, num_units[2], num_units[3], 3, padding=1) + \
           count_flops_conv(16, 16, num_units[3], num_units[4], 3, padding=1) # + \
           # count_flops_max_pool(16, 16, num_units[3], 3)

  flops += count_flops_conv(8, 8, num_units[4], num_units[5], 3, padding=1) + \
           count_flops_conv(8, 8, num_units[5], num_units[6], 3, padding=1) + \
           count_flops_conv(8, 8, num_units[6], num_units[7], 3, padding=1) # + \
           # count_flops_max_pool(8, 8, num_units[7], 2)

  flops += count_flops_conv(4, 4, num_units[7], num_units[8], 3, padding=1) + \
           count_flops_conv(4, 4, num_units[8], num_units[9], 3, padding=1) + \
           count_flops_conv(4, 4, num_units[9], num_units[10], 3, padding=1) # + \
           # count_flops_max_pool(4, 4, num_units[10], 2)

  flops += count_flops_conv(2, 2, num_units[10], num_units[11], 3, padding=1) + \
           count_flops_conv(2, 2, num_units[11], num_units[12], 3, padding=1) + \
           count_flops_conv(2, 2, num_units[12], num_units[13]/4, 3, padding=1)

  # flops += count_flops_avg_pool(2, 2, num_units[13])

  flops += count_flops_dense(num_units[13], num_units[14])
  flops += count_flops_dense(num_units[14], num_units[15])
  flops += count_flops_dense(num_units[15], n_classes)

  return flops


def resnet18_flops(num_units, n_classes):
  flops = count_flops_conv(32, 32, num_units[0], num_units[1], 3, padding=1)

  flops += count_flops_conv(32, 32, num_units[1], num_units[2], 3, padding=1) + \
           count_flops_conv(32, 32, num_units[2], num_units[3], 3, padding=1) + \
           count_flops_conv(32, 32, num_units[3], num_units[4], 3, padding=1) + \
           count_flops_conv(32, 32, num_units[4], num_units[5], 3, padding=1)

  flops += count_flops_conv(32, 32, num_units[5], num_units[6], 3, stride=2, padding=1) + \
           count_flops_conv(16, 16, num_units[6], num_units[8], 3, padding=1) + \
           count_flops_conv(32, 32, num_units[7], num_units[8], 1, stride=2, padding=0) + \
           count_flops_conv(16, 16, num_units[8], num_units[9], 3, padding=1) + \
           count_flops_conv(16, 16, num_units[9], num_units[10], 3, padding=1)

  flops += count_flops_conv(16, 16, num_units[10], num_units[11], 3, stride=2, padding=1) + \
           count_flops_conv(8, 8, num_units[11], num_units[13], 3, padding=1) + \
           count_flops_conv(16, 16, num_units[12], num_units[13], 1, stride=2, padding=0) + \
           count_flops_conv(8, 8, num_units[13], num_units[14], 3, padding=1) + \
           count_flops_conv(8, 8, num_units[14], num_units[15], 3, padding=1)

  flops += count_flops_conv(8, 8, num_units[15], num_units[16], 3, stride=2, padding=1) + \
           count_flops_conv(4, 4, num_units[16], num_units[18], 3, padding=1) + \
           count_flops_conv(8, 8, num_units[17], num_units[18], 1, stride=2, padding=0) + \
           count_flops_conv(4, 4, num_units[18], num_units[19], 3, padding=1) + \
           count_flops_conv(4, 4, num_units[19], num_units[20], 3, padding=1)

  # flops += count_flops_avg_pool(4, 4, num_units[20])
  flops += count_flops_dense(num_units[20], n_classes)

  return flops


if __name__ == '__main__':
    # Network Slim: [62, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 503, 448, 97, 'M', 24, 23, 289, 'M', 512]

    orig_cfg = [3, 64, 64, 64, 64, 64, 128, 64, 128, 128, 128, 256, 128, 256, 256, 256, 512, 256, 512, 512, 512]
    orig_flops = resnet18_flops(orig_cfg, 10)

    prune_cfg = [3, 63, 63, 63, 63, 63, 125, 63, 125, 125, 125, 250, 125, 250, 251, 251, 501, 251, 501, 501, 503]
    # prune_cfg = [3, 37, 58, 108, 112, 192, 106, 35, 48, 41, 16, 61, 63, 15, 15, 8]
    # prune_cfg = [3, 51, 53, 117, 120, 242, 233, 224, 383, 328, 304, 293, 307, 436]
    pruned_flops = resnet18_flops(prune_cfg, 10)

    print(f'Original Channel shape: {orig_cfg}')
    print(f'Pruned Channel shape: {prune_cfg}')
    print(f'Channel Sparsity: {1-np.sum(prune_cfg)/np.sum(orig_cfg)}')
    print(f'Compression rate in FLOPs: {pruned_flops}/{orig_flops} = {pruned_flops/orig_flops} (xFLOPs={orig_flops/pruned_flops})')
