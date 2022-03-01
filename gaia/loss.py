def total_variation(img):

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().mean(dim=reduce_axes)
    res2 = pixel_dif2.abs().mean(dim=reduce_axes)

    return res1 + res2