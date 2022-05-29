"""watermarking images"""
from PIL import Image
from numpy import asarray, array, zeros
from math import sqrt


class BlindImageWatermarking:
    def __init__(self, path_to_image='', path_to_watermark='', path_to_watermarked_image='', arnold_iterations=10,
                 dc_block_size=2, t=66, alpha=3, path_to_extracted_watermark=''):
        self.path_image = path_to_image
        self.path_watermark = path_to_watermark
        self.image_rgb_dict = {}
        self.watermark_rgb_dict = {}
        self.arnold_iterations = arnold_iterations
        self.dc_block_size = dc_block_size
        self.image_dc_matrix = {}
        self.image_dc_matrix_updated = {}
        self.t_dict = {'R': t * 0.78, 'G': t * 0.94, 'B': t}
        self.alpha = alpha
        self.watermark_sequence = {}
        self.path_to_watermarked_image = path_to_watermarked_image
        self.path_extracted_watermark = path_to_extracted_watermark

    def open_images(self, extracting):
        try:
            img = Image.open(self.path_image)
            if extracting:
                return img, ''
            watermark = Image.open(self.path_watermark)
            return img, watermark
        except Exception as e:
            print(e)
            return False

    def jpg_to_png(self, extracting=False) -> None:
        """Преобразует jpg к png"""
        img, watermark = self.open_images(extracting)
        path = self.path_image.split('.')
        path[-1] = '.png'
        self.path_image = ''.join(path)
        img.save(self.path_image)
        if extracting:
            return
        path = self.path_watermark.split('.')
        path[-1] = '.png'
        self.path_watermark = ''.join(path)
        watermark.save(self.path_watermark)

    def resize_512_512(self, extracting=False) -> None:
        """Преобразует изображение к размеру 512 * 512
           А водяной знак к размеру 32 * 32"""
        img, watermark = self.open_images(extracting)

        img_newsize = (512, 512)
        img = img.resize(img_newsize)
        img.save(self.path_image)
        if extracting:
            return

        watermark_newsize = (45, 45)
        watermark = watermark.resize(watermark_newsize)
        watermark.save(self.path_watermark)

    def image_rgb(self, extracting=False) -> None:
        """Получает RGB представляющие изображения"""
        img, watermark = self.open_images(extracting)
        for channel in 'RGB':
            img_one_channel = img.getchannel(channel)
            img_name = self.path_image.split('.')
            img_name[0] = f'{img_name[0]}_{channel}.'
            img_one_channel.save(''.join(img_name))
            self.image_rgb_dict[channel] = ''.join(img_name)

            watermark_one_channel = watermark.getchannel(channel)
            watermark_name = self.path_watermark.split('.')
            watermark_name[0] = f'{watermark_name[0]}_{channel}.'
            watermark_one_channel.save(''.join(watermark_name))
            self.watermark_rgb_dict[channel] = ''.join(watermark_name)

    def arnold_transform(self):
        """Arnold transform"""
        for channel in 'RGB':
            for _ in range(self.arnold_iterations):
                image = Image.open(self.watermark_rgb_dict[channel])
                dim = width, height = image.size
                transformed_watermark = Image.new(image.mode, dim)
                for x in range(width):
                    for y in range(height):
                        nx = (2 * x + y) % width
                        ny = (x + y) % height
                        transformed_watermark.putpixel((nx, height-ny-1), image.getpixel((x, height-y-1)))

                transformed_watermark.save(self.watermark_rgb_dict[channel])

    def arnold_transform_back(self):
        """Arnold transform back"""
        for channel in 'RGB':
            for _ in range(self.arnold_iterations):
                image = Image.open(self.watermark_rgb_dict[channel])
                dim = width, height = image.size
                transformed_watermark = Image.new(image.mode, dim)
                for x in range(width):
                    for y in range(height):
                        nx = (x - y) % width
                        ny = (-x + 2 * y) % height
                        transformed_watermark.putpixel((nx, height - ny - 1), image.getpixel((x, height - y - 1)))

                transformed_watermark.save(self.watermark_rgb_dict[channel])

    def dc_matrix(self):
        """Calculate DC coefficient matrix"""
        for channel, im in self.image_rgb_dict.items():
            image = Image.open(im)
            image_array = asarray(image)
            matrix_size = image.size[0] // self.dc_block_size
            matrix = zeros((matrix_size, matrix_size))
            for i in range(0, image.size[0], self.dc_block_size):
                for j in range(0, image.size[1], self.dc_block_size):
                    dc = 0
                    for x in range(self.dc_block_size):
                        for y in range(self.dc_block_size):
                            dc += image_array[i+y][j+x]
                    dc /= self.dc_block_size
                    matrix[i // self.dc_block_size, j // self.dc_block_size] = dc
            self.image_dc_matrix[channel] = matrix

    def watermark_bit_sequence(self):
        """To obtain the layered watermark bit sequence"""
        for channel, im in self.watermark_rgb_dict.items():
            sequence = ''
            image = Image.open(im)
            image_array = asarray(image)
            for line in image_array:
                sequence += ''.join([f'{pix:08b}' for pix in line])
            self.watermark_sequence[channel] = sequence

    def embedding_watermark(self):
        """Embedding watermark bits and obtaining the watermarked DC coefficient matrix DC-map*"""
        matrix_size = len(self.image_dc_matrix['R'])
        for channel, sequence in self.watermark_sequence.items():
            t = self.t_dict[channel]
            dc_updated = self.image_dc_matrix[channel].copy()
            x = 0
            for bit in sequence:
                dh = 0
                for i in range(self.dc_block_size):
                    for j in range(self.dc_block_size):
                        dh += self.image_dc_matrix[channel][(x // matrix_size) * self.dc_block_size + i, x % matrix_size + j]
                if bit == '1':
                    dh_new = self.alpha * t * round(dh / (self.alpha * t)) + t / 2
                else:
                    dh_new = self.alpha * t * round(dh / (self.alpha * t)) - t / 2
                for i in range(self.dc_block_size):
                    for j in range(self.dc_block_size):
                        dc_updated[(x // matrix_size) * self.dc_block_size + i, x % matrix_size + j] = \
                            self.image_dc_matrix[channel][(x // matrix_size) * self.dc_block_size + i, x % matrix_size + j] + \
                                               (dh_new - dh)/(self.dc_block_size ** 2)

                x += self.dc_block_size
            self.image_dc_matrix_updated[channel] = dc_updated

    def update_pixels_to_watermarked(self):
        """Update pixels in layered carrier image to watermarked pixel's values"""
        for channel, im in self.image_rgb_dict.items():
            image = Image.open(im)
            image_array = asarray(image)
            m = self.dc_block_size
            for i in range(0, image.size[0], self.dc_block_size):
                for j in range(0, image.size[1], self.dc_block_size):
                    new_pixel = (self.image_dc_matrix_updated[channel][i//m][j//m] -
                                 self.image_dc_matrix[channel][i//m][j//m]) / m
                    for y in range(m):
                        for x in range(m):
                            new_pixel_value = new_pixel + image_array[i + y][j + x]
                            image_array[i + y][j + x] = new_pixel_value
            watermarked_image = Image.fromarray(image_array)
            watermarked_image.save(im)

    def image_merge(self):
        """Combine the three-layered watermarked image to obtain watermarked image"""
        img_r = Image.open(self.image_rgb_dict['R'])
        img_g = Image.open(self.image_rgb_dict['G'])
        img_b = Image.open(self.image_rgb_dict['B'])
        my = Image.merge('RGB', (img_r, img_g, img_b))
        my.save('images/result.png')
        self.path_to_watermarked_image = 'images/result.png'

    def extracting_watermarks_bits(self):
        """Getting watermark's bit sequence"""
        bit_sequence = Image.open(self.path_watermark)
        h, w = bit_sequence.size
        len_bit_sequence = h * w * 8
        dc_size = len(self.image_dc_matrix['G'])
        for channel, matrix in self.image_dc_matrix.items():
            sequence = ''
            for num_bit in range(len_bit_sequence):
                dh = 0
                for i in range(self.dc_block_size):
                    for j in range(self.dc_block_size):
                        dh += matrix[(num_bit * self.dc_block_size // dc_size) * self.dc_block_size + i,
                                     num_bit * self.dc_block_size % dc_size + j]
                if dh - self.alpha * self.t_dict[channel] * round(dh / (self.alpha * self.t_dict[channel])) > 0:
                    sequence += '1'
                else:
                    sequence += '0'
            self.watermark_sequence[channel] = sequence

    def sequence_to_watermark(self):
        """Transform bit sequence into layered watermark"""
        bit_sequence = Image.open(self.path_watermark)
        h, w = bit_sequence.size
        for channel, sequence in self.watermark_sequence.items():
            watermark_list = []
            for i in range(h):
                watermark_list.append([int(sequence[i * w * 8 + j * 8:i * w * 8 + j * 8 + 8], 2) for j in range(w)])
            watermark_list = array(watermark_list)
            watermark = Image.fromarray(watermark_list).convert('L')
            watermark.save(self.watermark_rgb_dict[channel])

    def merge_watermark(self):
        """Combine the three-layered watermark to obtain color watermark image"""
        wat_r = Image.open(self.watermark_rgb_dict['R'])
        wat_g = Image.open(self.watermark_rgb_dict['G'])
        wat_b = Image.open(self.watermark_rgb_dict['B'])
        my = Image.merge('RGB', (wat_r, wat_g, wat_b))
        my.save('images/result_watermark.png')
        self.path_extracted_watermark = 'images/result_watermark.png'

    def watermarking(self):
        """Watermarking procedure"""
        assert self.path_image, 'Для нанесения водяного знака передайте путь до изображения'
        assert self.path_watermark, 'Для нанесения водяного знака передайте путь до встраиваемого изображения'
        self.resize_512_512()
        self.image_rgb()
        self.arnold_transform()
        self.dc_matrix()
        self.watermark_bit_sequence()
        self.embedding_watermark()
        self.update_pixels_to_watermarked()
        self.image_merge()

    def watermark_extracting(self):
        """Watermark extracting procedure"""
        assert self.path_to_watermarked_image, 'Не указан путь до изображения содержащего водяной знак'
        assert self.path_watermark, 'Не указан путь до изображения водяного знака'
        self.path_image = self.path_to_watermarked_image
        self.jpg_to_png()
        self.resize_512_512(extracting=True)
        self.image_rgb()
        self.dc_matrix()
        self.extracting_watermarks_bits()
        self.sequence_to_watermark()
        self.arnold_transform_back()
        self.merge_watermark()

    def normalized_correlation(self):
        """Calculation correlation coefficient"""
        assert self.path_watermark, 'Не указан путь до изображения водяного знака'
        assert self.path_extracted_watermark, 'Не указан путь до извлеченного водяного знака'
        watermark = Image.open(self.path_watermark)
        watermark_ext = Image.open(self.path_extracted_watermark)
        width, height = watermark.size
        # Dividing the formula into elements a, b and c -> a/sqrt(b * sqrt (c))
        a, b, c = 0, 0, 0
        for channel in 'RGB':
            wat = watermark.getchannel(channel)
            wat_ext = watermark_ext.getchannel(channel)
            for u in range(height):
                for v in range(width):
                    pixel_w = wat.getpixel((u, v))
                    pixel_w_e = wat_ext.getpixel((u, v))
                    a += pixel_w * pixel_w_e
                    b += pixel_w ** 2
                    c += pixel_w_e ** 2
        nc = a / (sqrt(b) * sqrt(c))
        return nc


my_img = BlindImageWatermarking('images/cute_cat.png', 'images/hse.png')
my_img.watermarking()

my_img.watermark_extracting()




