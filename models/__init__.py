def IS2D_model(device, model_name, image_size, num_channels):
    if model_name == 'M3FPolypSegNet':
        from models.m3fpolypsegnet import M3FPolypSegNet
        return M3FPolypSegNet(device, num_channels, image_size, 32)