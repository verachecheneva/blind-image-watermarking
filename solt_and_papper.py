import random
from PIL import Image
from numpy import asarray


def add_noise(img):
    row, col = img.size
    img = asarray(img)
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    img = Image.fromarray(img)
    return img


# img = Image.open('images/for_checks_result.png')
# img_dict = {}
# for channel in 'RGB':
#     img_new = img.getchannel(channel)
#     img_new = add_noise(img_new)
#     img_dict[channel] = img_new
#
# my = Image.merge('RGB', (img_dict['R'], img_dict['G'], img_dict['B']))
# my.save('images/result.png')