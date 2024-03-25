import random

from tmu.composite.components.base import TMComponent
import cv2
import numpy as np


class RandomPatchExtractionComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, 24, 24, ch), dtype=np.uint8)
        patch_size = 24

        for s in range(samples):
            start_x = np.random.randint(0, rows - patch_size)
            start_y = np.random.randint(0, cols - patch_size)
            processed_X[s] = data["X"][s, start_x:start_x + patch_size, start_y:start_y + patch_size, :]

        return {'X': processed_X, 'Y': data["Y"]}


class SharpeningComponent(TMComponent):
    def preprocess(self, data: dict):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            processed_X[s] = cv2.filter2D(data["X"][s], -1, kernel)

        return {'X': processed_X, 'Y': data["Y"]}


class ChannelNormalizationComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            for i in range(3):  # assuming RGB channels
                processed_X[s, ..., i] = (data["X"][s, ..., i] - np.mean(data["X"][s, ..., i])) / (
                        np.std(data["X"][s, ..., i]) + 1e-7)

        return {'X': processed_X, 'Y': data["Y"]}


class SaltAndPepperNoiseComponent(TMComponent):
    def preprocess(self, data: dict):
        noise_amt = 0.02
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.copy(data["X"])

        for s in range(samples):
            total_pixels = rows * cols

            num_salt = np.ceil(noise_amt * total_pixels)
            num_pepper = np.ceil(noise_amt * total_pixels)

            coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in [rows, cols]]
            processed_X[s, coords_salt[0], coords_salt[1], :] = 255

            coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in [rows, cols]]
            processed_X[s, coords_pepper[0], coords_pepper[1], :] = 0

        return {'X': processed_X, 'Y': data["Y"]}


class MorphologicalOperationsComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)

        for s in range(samples):
            processed_X[s] = cv2.morphologyEx(data["X"][s], cv2.MORPH_CLOSE, kernel)

        return {'X': processed_X, 'Y': data["Y"]}


class InvertColorsComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            processed_X[s] = 255 - data["X"][s]

        return {'X': processed_X, 'Y': data["Y"]}


class GammaCorrectionComponent(TMComponent):
    def preprocess(self, data: dict):
        gamma = 2.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            processed_X[s] = cv2.LUT(data["X"][s], table)

        return {'X': processed_X, 'Y': data["Y"]}


class OpticalFlowComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        # Iterate through the batch
        for s in range(samples):
            # For the first image, use it as both current and previous to get zero flow.
            prev_img = data["X"][s - 1] if s > 0 else data["X"][s]

            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY),
                                                cv2.cvtColor(data["X"][s], cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5,
                                                1.2, 0)
            hsv = np.zeros((rows, cols, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            processed_X[s] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return {'X': processed_X, 'Y': data["Y"]}


class RandomRescaleComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            factor = random.uniform(0.8, 1.2)
            rescaled = cv2.resize(data["X"][s], None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

            if factor > 1:  # Image is larger and needs to be cropped
                start_x = random.randint(0, rescaled.shape[1] - cols)
                start_y = random.randint(0, rescaled.shape[0] - rows)
                rescaled = rescaled[start_y:start_y + rows, start_x:start_x + cols]
            elif factor < 1:  # Image is smaller and needs to be padded
                pad_x = (cols - rescaled.shape[1]) // 2
                pad_y = (rows - rescaled.shape[0]) // 2
                rescaled = cv2.copyMakeBorder(rescaled, pad_y, rows - rescaled.shape[0] - pad_y,
                                              pad_x, cols - rescaled.shape[1] - pad_x,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])  # Using black for padding

            processed_X[s] = rescaled

        return {'X': processed_X, 'Y': data["Y"]}



class CLAHEComponent:
    def preprocess(self, data: dict):
        # Extracting shape details
        samples, rows, cols, ch = data["X"].shape

        # Pre-allocation of the output array for efficiency
        processed_X = np.empty((samples, rows, cols, ch), dtype=data["X"].dtype)

        # Initializing CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Applying CLAHE to each sample and each channel
        for s in range(samples):
            for i in range(ch):
                processed_X[s, ..., i] = clahe.apply(data["X"][s, ..., i])

        return {'X': processed_X, 'Y': data["Y"]}


class RandomTranslationComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            tx, ty = random.randint(-10, 10), random.randint(-10, 10)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            processed_X[s] = cv2.warpAffine(data["X"][s], M, (cols, rows))

        return {'X': processed_X, 'Y': data["Y"]}


class RandomRotation90Component(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            k = random.randint(0, 3)
            processed_X[s] = np.rot90(data["X"][s], k)

        return {'X': processed_X, 'Y': data["Y"]}


class EdgeDetectionComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            edges = cv2.Canny(data["X"][s], 100, 200)
            processed_X[s] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return {'X': processed_X, 'Y': data["Y"]}


class GaborFilterComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        for s in range(samples):
            processed_X[s] = cv2.filter2D(data["X"][s], cv2.CV_8UC3, kernel)

        return {'X': processed_X, 'Y': data["Y"]}


class BilateralFilterComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            processed_X[s] = cv2.bilateralFilter(data["X"][s], 9, 75, 75)

        return {'X': processed_X, 'Y': data["Y"]}


class DCTComponent:
    def preprocess(self, data: dict):
        # Extracting shape details
        samples, rows, cols, ch = data["X"].shape

        # Pre-allocation of the output array for efficiency
        processed_X = np.empty((samples, rows, cols, ch), dtype=data["X"].dtype)

        # Applying DCT to each channel of each sample
        for s in range(samples):
            for c in range(ch):
                processed_X[s, ..., c] = cv2.dct(data["X"][s, ..., c])

        return {'X': processed_X, 'Y': data["Y"]}


class HueSaturationValueAdjustmentComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-40, 40), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-40, 40), 0, 255)
            processed_X[s] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return {'X': processed_X, 'Y': data["Y"]}


class ColorBalanceComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            for i in range(3):
                img[..., i] = np.clip(img[..., i] + random.randint(-20, 20), 0, 255)
            processed_X[s] = img

        return {'X': processed_X, 'Y': data["Y"]}


class GradientMagnitudeComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            processed_X[s] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {'X': processed_X, 'Y': data["Y"]}


class FFTComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            f_transform = np.fft.fft2(img, axes=(0, 1))
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
            processed_X[s] = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {'X': processed_X, 'Y': data["Y"]}


class GaussianBlurComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            processed_X[s] = cv2.GaussianBlur(img, (5, 5), 0)

        return {'X': processed_X, 'Y': data["Y"]}


class ImageRotationComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            processed_X[s] = cv2.warpAffine(img, M, (cols, rows))

        return {'X': processed_X, 'Y': data["Y"]}




class HorizontalFlipComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            processed_X[s] = cv2.flip(img, 1)

        return {'X': processed_X, 'Y': data["Y"]}


class VerticalFlipComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            processed_X[s] = cv2.flip(img, 0)

        return {'X': processed_X, 'Y': data["Y"]}


class BrightnessAdjustmentComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            factor = random.uniform(0.5, 1.5)
            processed_X[s] = np.clip(img * factor, 0, 255).astype(np.uint8)

        return {'X': processed_X, 'Y': data["Y"]}


class ContrastAdjustmentComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            factor = random.uniform(0.5, 1.5)
            processed_X[s] = np.clip(127.5 + factor * (img - 127.5), 0, 255).astype(np.uint8)

        return {'X': processed_X, 'Y': data["Y"]}


class ImageNormalizationComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.float32)

        for s in range(samples):
            img = data["X"][s]
            processed_X[s] = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'X': processed_X, 'Y': data["Y"]}


class ImageShiftComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            shift_x = random.randint(-10, 10)
            shift_y = random.randint(-10, 10)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            processed_X[s] = cv2.warpAffine(img, M, (cols, rows))

        return {'X': processed_X, 'Y': data["Y"]}


class ChannelShuffleComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            channels = [0, 1, 2]
            random.shuffle(channels)
            processed_X[s] = img[..., channels]

        return {'X': processed_X, 'Y': data["Y"]}


class AffineTransformationComponent(TMComponent):

    def preprocess(self, data: dict):
        # Extracting shape details
        samples, rows, cols, ch = data["X"].shape

        # Pre-allocation of the output array for efficiency
        processed_X = np.empty((samples, rows, cols, ch), dtype=data["X"].dtype)

        # Defining transformation matrices
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)

        # Applying the transformation to each sample
        for i in range(samples):
            processed_X[i] = cv2.warpAffine(data["X"][i], M, (cols, rows))

        return {'X': processed_X, 'Y': data["Y"]}


class PerspectiveTransformComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        for s in range(samples):
            img = data["X"][s]
            processed_X[s] = cv2.warpPerspective(img, M, (cols, rows))

        return {'X': processed_X, 'Y': data["Y"]}


class NoiseInjectionComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            noise = np.random.randn(*img.shape) * 25  # Adjust noise level
            processed_X[s] = np.clip(img + noise, 0, 255).astype(np.uint8)

        return {'X': processed_X, 'Y': data["Y"]}


class HistogramEqualizationComponent(TMComponent):
    def preprocess(self, data: dict):
        samples, rows, cols, ch = data["X"].shape
        processed_X = np.empty((samples, rows, cols, ch), dtype=np.uint8)

        for s in range(samples):
            img = data["X"][s]
            for i in range(3):
                processed_X[s, ..., i] = cv2.equalizeHist(img[..., i])

        return {'X': processed_X, 'Y': data["Y"]}


class ColorJitterComponent:
    def preprocess(self, data: dict):
        # Extracting shape details
        samples, rows, cols, ch = data["X"].shape

        # Pre-allocation of the output array for efficiency
        processed_X = np.empty((samples, rows, cols, ch), dtype=data["X"].dtype)

        # Applying color jittering to each sample
        for s in range(samples):
            img = data["X"][s]

            # Brightness, Contrast, Saturation adjustment
            brightness = random.uniform(0.5, 1.5)
            contrast = random.uniform(0.5, 1.5)
            saturation = random.uniform(0.5, 1.5)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            img = np.clip(127.5 + contrast * (img - 127.5), 0, 255).astype(np.uint8)

            processed_X[s] = img

        return {'X': processed_X, 'Y': data["Y"]}
