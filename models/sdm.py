import torch
    
def On_attention_gaussian_mask(num_patches):
    mask = torch.zeros(num_patches, num_patches)
    img_size = num_patches ** 0.5
    for i in range(num_patches):
        for j in range(num_patches):
            x_change = i % img_size - j % img_size
            y_change = i // img_size - j // img_size
            mask[i, j] = - (x_change ** 2 + y_change ** 2)
    mask = torch.exp(mask) ** (1 / 2)
    return mask.expand(1, 1, num_patches, num_patches)

