def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
"""
    for i in range(len(out_pool_size)):
        # out_pool_size是一个数组，例如[1,2,4],表示需要将原先的特征图分别划分为1×1、2×2、4×4三种.
        h, w = previous_conv_size
        # h,w表示原先特征图的长和宽
        h_wid = math.ceil(h / out_pool_size[i])
        w_wid = math.ceil(w / out_pool_size[i])
        # 计算每一个子块的长和宽，这里需要进行取整
        h_str = math.floor(h / out_pool_size[i])
        w_str = math.floor(w / out_pool_size[i])
        # 计算池化的步长
        max_pool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_str, w_str))
        x = max_pool(previous_conv)
        print(x.shape)
        print(x.view(num_sample, -1).shape)
        # 对每个子块进行最大池化
        if i == 0:
            spp = x.view(num_sample, -1)
#             print(spp.shape)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
#             print(spp.shape)
     # 拼接各个子块的输出结果
    return spp