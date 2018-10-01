from torchvision.transforms import *
import torch
from PIL import Image
import random
import numpy as np
import scipy.ndimage as ndi
from math import floor


class RandomZoom(object):
    def __init__(self, prob=0.0, zoom_range=[1, 1]):
        self.prob = prob
        self.zoom_range = zoom_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                        resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) -
                                            (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) +
                                            (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))


class RandomStretch(object):
    def __init__(self, prob=0.0, stretch_range=[1, 1]):
        self.prob = prob
        self.stretch_range = stretch_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor = random.uniform(
                self.stretch_range[0], self.stretch_range[1])
            if factor <= 1:
                image_stretched = image.resize((int(round(image.size[0] / factor)),
                                                int(round(image.size[1]))),
                                               resample=Image.BICUBIC)
            else:
                image_stretched = image.resize((int(round(image.size[0])),
                                                int(round(image.size[1] * factor))),
                                               resample=Image.BICUBIC)
            w_stretched, h_stretched = image_stretched.size

            return image_stretched.crop((floor((float(w_stretched) / 2) - (float(w) / 2)),
                                         floor((float(h_stretched) / 2) -
                                               (float(h) / 2)),
                                         floor((float(w_stretched) / 2) +
                                               (float(w) / 2)),
                                         floor((float(h_stretched) / 2) + (float(h) / 2))))


class RandomResize(object):
    def __init__(self, prob=0.0, resize_range=[1, 1]):
        self.prob = prob
        self.resize_range = resize_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor_w = random.uniform(
                self.resize_range[0], self.resize_range[1])
            factor_h = random.uniform(
                self.resize_range[0], self.resize_range[1])
            image_resized = image.resize((int(round(image.size[0] * factor_w)),
                                          int(round(image.size[1] * factor_h))),
                                         resample=Image.BICUBIC)
            w_resized, h_resized = image_resized.size

            return image_resized.crop((floor((float(w_resized) / 2) - (float(w) / 2)),
                                       floor((float(h_resized) / 2) -
                                             (float(h) / 2)),
                                       floor((float(w_resized) / 2) +
                                             (float(w) / 2)),
                                       floor((float(h_resized) / 2) + (float(h) / 2))))


class RandomRotation(object):
    def __init__(self, prob=0.0, degree=[0, 0]):
        self.prob = prob
        self.degree = degree

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            rotate_degree = random.uniform(self.degree[0], self.degree[1])
            rotated_array = ndi.interpolation.rotate(
                image, rotate_degree, reshape=False, mode='nearest')
            rotated_img = Image.fromarray(rotated_array)

            return rotated_img


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img
        else:
            return functional.hflip(img)


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img
        else:
            return functional.vflip(img)


def RotateTensor(input, degree):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    RandomRotation(prob=1.0, degree=degree),
                                    transforms.ToTensor()])
    return transform(input)


def ZoomTensor(input, factor):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    RandomZoom(prob=1.0, zoom_range=factor),
                                    transforms.ToTensor()])
    return transform(input)


def FlipTensor(input, fclass):
    if fclass == 1:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor()])
    elif fclass == 2:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        RandomHorizontalFlip(
                                            prob=1.0),
                                        transforms.ToTensor()])
    elif fclass == 3:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        RandomHorizontalFlip(prob=1.0),
                                        RandomVerticalFlip(prob=1.0),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        RandomVerticalFlip(prob=1.0),
                                        transforms.ToTensor()])
    return transform(input)


# def RelabeledRandomRotation(input, label, num_rotation_class=1, prob=0.0):
#     batch_size = label.size()[0]
#     interval = 360 / num_rotation_class
#     for i in range(batch_size):
#         if random.uniform(0, 1) <= prob:
#             rotation_class = random.randint(1, num_rotation_class)
#             label[i] = label[i] * num_rotation_class + rotation_class - 1
#             degree = (interval * (rotation_class - 1), interval * rotation_class)
#             rotated_image = RotateTensor(input[i], degree)
#             input[i] = rotated_image
#     return input, label

def RelabeledRandomRotation(input, label, num_rotation_class=1):
    input_new = input.clone()
    label_new = label.clone()
    print(label_new.size()[0])
    batch_size = label.size()[0]
    interval = 360 / num_rotation_class
    for i in range(batch_size):
        rotation_class = random.randint(1, num_rotation_class)
        label_new[i] = label[i] * num_rotation_class + rotation_class - 1
        degree = [interval * (rotation_class - 1), interval * rotation_class]
        rotated_image = RotateTensor(input[i], degree)
        input_new[i] = rotated_image
    return input_new, label_new


def AuxRandomRotation(input, num_rotation_class=1, degree_range=360):
    batch_size = input.size()[0]
    label_aux = torch.LongTensor(batch_size).zero_()
    interval = degree_range / num_rotation_class
    for i in range(batch_size):
        rotation_class = random.randint(1, num_rotation_class)
        label_aux[i] = rotation_class - 1
        degree = [interval * (rotation_class - 1), interval * rotation_class]
        rotated_image = RotateTensor(input[i], degree)
        input[i] = rotated_image
    return input, label_aux


def AuxRandomRotationLinear(input, degree_range=360):
    input_new = input.clone()
    batch_size = input.size()[0]
    label_new = torch.FloatTensor(batch_size).zero_()
    for i in range(batch_size):
        degree = np.random.uniform(0, degree_range)
        label_new[i] = degree
        rotated_image = RotateTensor(input[i], [degree, degree])
        input_new[i] = rotated_image
    return input_new, label_new


def AuxRandomRotationSample(input,  prob,
                            num_rotation_class=1, degree_range=360):
    [batch_size, channel, width, height] = input.size()
    do_rotation = (np.random.rand(batch_size) < prob)
    num_rotated = do_rotation.sum()
    label_rot = torch.LongTensor(num_rotated).zero_()
    input_rot = torch.FloatTensor(num_rotated, channel, width, height).zero_()

    interval = degree_range / num_rotation_class
    rotated_counter = 0
    for i in range(batch_size):
        if do_rotation[i]:
            rotation_class = random.randint(1, num_rotation_class)
            label_rot[rotated_counter] = rotation_class - 1
            degree = [interval * (rotation_class - 1),
                      interval * rotation_class]
            rotated_image = RotateTensor(input[i], degree)
            input_rot[rotated_counter] = rotated_image
            input[i] = rotated_image
            rotated_counter += 1

    return input_rot, label_rot


def AuxRandomRotationSample_2(input, label, prob,
                              num_rotation_class=1, degree_range=360):
    [batch_size, channel, width, height] = input.size()
    do_rotation = (np.random.rand(batch_size) < prob)
    num_rotated = do_rotation.sum()
    label_rot = torch.LongTensor(num_rotated).zero_()
    label_rot_n = torch.LongTensor(num_rotated).zero_()
    input_rot = torch.FloatTensor(num_rotated, channel, width, height).zero_()
    label_unrot = torch.LongTensor(batch_size - num_rotated).zero_()
    input_unrot = torch.FloatTensor(batch_size - num_rotated,
                                    channel, width, height).zero_()

    interval = degree_range / num_rotation_class
    rotated_counter = 0
    unrot_counter = 0
    for i in range(batch_size):
        if do_rotation[i]:
            rotation_class = random.randint(1, num_rotation_class)
            label_rot[rotated_counter] = rotation_class - 1
            degree = [interval * (rotation_class - 1),
                      interval * rotation_class]
            rotated_image = RotateTensor(input[i], degree)
            input_rot[rotated_counter] = rotated_image
            label_rot_n[rotated_counter] = label[i]
            rotated_counter += 1
        else:
            input_unrot[unrot_counter] = input[i]
            label_unrot[unrot_counter] = label[i]
            unrot_counter += 1
    assert(rotated_counter + unrot_counter == batch_size)

    return input_unrot, label_unrot, input_rot, label_rot_n, label_rot


def AuxRandomRotationReserved(input,  prob,
                              num_rotation_class=1, degree_range=360):
    [batch_size, channel, width, height] = input.size()
    do_rotation = (np.random.rand(batch_size) < prob)
    num_rotated = do_rotation.sum()
    label_rot = torch.LongTensor(num_rotated).zero_()
    input_rot = torch.FloatTensor(num_rotated, channel, width, height).zero_()

    interval = degree_range / num_rotation_class
    rotated_counter = 0
    for i in range(batch_size):
        if do_rotation[i]:
            rotation_class = random.randint(1, num_rotation_class)
            label_rot[rotated_counter] = rotation_class - 1
            degree = [interval * (rotation_class - 1),
                      interval * rotation_class]
            rotated_image = RotateTensor(input[i], degree)
            input_rot[rotated_counter] = rotated_image
            rotated_counter += 1

    return input_rot, label_rot


def AuxRandomZoomSample(input,  prob,
                        num_zoom_class=1, zoom_range=[0.7, 1.4]):
    [batch_size, channel, width, height] = input.size()
    do_zoom = (np.random.rand(batch_size) < prob)
    num_zoomed = do_zoom.sum()
    label_zoom = torch.LongTensor(num_zoomed).zero_()
    input_zoom = torch.FloatTensor(num_zoomed, channel, width, height).zero_()

    interval = (zoom_range[1] - zoom_range[0]) / num_zoom_class
    zoom_counter = 0
    for i in range(batch_size):
        if do_zoom[i]:
            zoom_class = random.randint(1, num_zoom_class)
            label_zoom[zoom_counter] = zoom_class - 1
            factor = [interval * (zoom_class - 1) + zoom_range[0],
                      interval * zoom_class + zoom_range[0]]
            zoomed_image = ZoomTensor(input[i], factor)
            input_zoom[zoom_counter] = zoomed_image
            input[i] = zoomed_image
            zoom_counter += 1

    return input_zoom, label_zoom


def AuxRandomRotationAll(input, prob, num_rotation_class=1, degree_range=360):
    [batch_size, channel, width, height] = input.size()
    do_rotation = (np.random.rand(batch_size) < prob)
    label_rot = torch.LongTensor(batch_size).zero_()

    interval = degree_range / num_rotation_class
    for i in range(batch_size):
        if do_rotation[i]:
            rotation_class = random.randint(1, num_rotation_class)
            label_rot[i] = rotation_class - 1
            degree = [interval * (rotation_class - 1),
                      interval * rotation_class]
            rotated_image = RotateTensor(input[i], degree)
            input[i] = rotated_image

    return input, label_rot


def AuxRandomRotationDiscrete(input, prob, num_rotation_class=1, degree_range=360):
    [batch_size, channel, width, height] = input.size()
    do_rotation = (np.random.rand(batch_size) < prob)
    label_rot = torch.LongTensor(batch_size).zero_()

    interval = degree_range / num_rotation_class
    for i in range(batch_size):
        if do_rotation[i]:
            rotation_class = random.randint(1, num_rotation_class)
            label_rot[i] = rotation_class - 1
            degree = [interval * (rotation_class - 1),
                      interval * (rotation_class - 1)]
            rotated_image = RotateTensor(input[i], degree)
            input[i] = rotated_image

    return input, label_rot


def AuxRandomZoom(input, num_zoom_class=1, zoom_range=[0.7, 1.4]):
    input_new = input.clone()
    batch_size = input.size()[0]
    label_new = torch.LongTensor(batch_size).zero_()
    interval = (zoom_range[1] - zoom_range[0]) / num_zoom_class
    for i in range(batch_size):
        zoom_class = random.randint(1, num_zoom_class)
        label_new[i] = zoom_class - 1
        factor = [interval * (zoom_class - 1) + zoom_range[0],
                  interval * zoom_class + zoom_range[0]]
        zoomed_image = ZoomTensor(input[i], factor)
        input_new[i] = zoomed_image
    return input_new, label_new


def AuxRandomRotationBound(input, num_rotation_class=1):
    input_new = input.clone()
    batch_size = input.size()[0]
    label_new = torch.LongTensor(batch_size).zero_()
    interval = 360 / num_rotation_class
    for i in range(batch_size):
        rotation_class = random.randint(1, num_rotation_class)
        label_new[i] = rotation_class - 1
        degree = [interval * rotation_class -
                  10, interval * rotation_class + 10]
        rotated_image = RotateTensor(input[i], degree)
        input_new[i] = rotated_image
    return input_new, label_new


def AuxRandomFlip(input):
    input_new = input.clone()
    batch_size = input.size()[0]
    label_new = torch.LongTensor(batch_size).zero_()
    for i in range(batch_size):
        flip_class = random.randint(1, 4)
        label_new[i] = flip_class - 1
        flipped_image = FlipTensor(input[i], flip_class)
        input_new[i] = flipped_image
    return input_new, label_new


def HorizontalFlip(input):
    batch_size = input.size()[0]
    for i in range(batch_size):
        flip_class = random.randint(1, 2)
        input[i] = FlipTensor(input[i], flip_class)
    return input
