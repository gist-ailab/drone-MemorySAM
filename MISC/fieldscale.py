"""
Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image
Source: https://github.com/HyeonJaeGil/fieldscale

@article{gil2024fieldscale,
  title={Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image},
  author={Gil, Hyeonjae and Jeon, Myung-Hwan and Kim, Ayoung},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
"""
import numpy as np
import cv2
from typing import Literal, Union


def gridwise_min(image: np.ndarray, grid_shape: tuple = (1, 1)) -> np.ndarray:
    """Return the minimum value of each patch in the image."""
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output = np.zeros(grid_shape, dtype=image.dtype)
    for i, j in np.ndindex(grid_shape):
        output[i, j] = np.amin(image[
            patch_shape[0] * i : patch_shape[0] * (i + 1),
            patch_shape[1] * j : patch_shape[1] * (j + 1)
        ])
    return output


def gridwise_max(image: np.ndarray, grid_shape: tuple = (1, 1)) -> np.ndarray:
    """Return the maximum value of each patch in the image."""
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output = np.zeros(grid_shape, dtype=image.dtype)
    for i, j in np.ndindex(grid_shape):
        output[i, j] = np.amax(image[
            patch_shape[0] * i : patch_shape[0] * (i + 1),
            patch_shape[1] * j : patch_shape[1] * (j + 1)
        ])
    return output


def get_neighbor_grids(grid: np.ndarray, xy: tuple, max_distance: int = 1) -> list:
    """Return the neighbors of a pixel in a grid."""
    h, w = grid.shape
    x, y = xy
    neighbors = [
        (x + i, y + j)
        for i in range(-max_distance, max_distance + 1)
        for j in range(-max_distance, max_distance + 1)
        if (i != 0 or j != 0) and (0 <= x + i < h) and (0 <= y + j < w)
    ]
    return sorted(neighbors, key=lambda k: (k[0], k[1]))


def local_extrema_suppression(grid: np.ndarray,
                              local_distance: int,
                              diff_threshold: float,
                              extrema: Literal["max", "min"]) -> np.ndarray:
    """Clip the extreme values in the grid."""
    assert extrema in ["max", "min"]
    if local_distance <= 0 or diff_threshold <= 0:
        return grid

    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=local_distance)
        neighbor_values = np.array([grid[xy] for xy in neighbors])
        if extrema == "max" and grid[i, j] >= neighbor_values.max():
            diff = grid[i, j] - neighbor_values.mean()
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() + diff_threshold
        elif extrema == "min" and grid[i, j] <= neighbor_values.min():
            diff = neighbor_values.mean() - grid[i, j]
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() - diff_threshold
    return grid


def message_passing(grid: np.ndarray,
                    direction: Literal["increase", "decrease"]) -> np.ndarray:
    """Message passing algorithm for grid."""
    assert direction in ["increase", "decrease"]
    grid_new = np.zeros_like(grid, dtype=np.float64)

    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=1)
        neighbors_value = [grid[neighbor] for neighbor in neighbors]
        mean = np.mean(neighbors_value + [grid[i, j]])
        bigger, smaller = (mean, grid[i, j]) if mean > grid[i, j] else (grid[i, j], mean)
        grid_new[i, j] = bigger if direction == "increase" else smaller

    return grid_new


def rescale_image_with_fields(image: np.ndarray,
                              min_field: np.ndarray,
                              max_field: np.ndarray,
                              min_range: float = 5.0) -> np.ndarray:
    """Rescale: (image - min_field) / (max_field - min_field) * 255.

    min_range: 분모의 하한. (고대비 vs 폭발 방지는 서로 반대 세팅)
    - 고대비 세팅: min_range 작게 (1~5) → 나눗셈으로 대비 극대화 → 좁은 구역에서 255 폭발 위험.
    - 폭발 방지 세팅: min_range 크게 (10~20) → 나눗셈 완화 → 하양 폭발 없음, 대신 어둡고 평평해짐.
    둘 다 만족하려면: 입력 전에 좁은 범위(90~99)를 0~255로 스트레칭해 두고, min_range는 5~10으로 절충.
    """
    assert image.shape == min_field.shape == max_field.shape

    image = image.astype(np.float64)
    min_field = min_field.astype(np.float64)
    max_field = max_field.astype(np.float64)

    min_field = np.where(min_field > max_field, max_field, min_field)
    max_field = np.where(max_field < min_field, min_field, max_field)
    image = np.clip(image, min_field, max_field)
    range_ = max_field - min_field
    safe_range = np.maximum(range_, min_range)
    image = (image - min_field) / safe_range * 255
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


class Fieldscale:
    def __init__(self, max_diff: float = 400, min_diff: float = 400,
                 iteration: int = 7, gamma: float = 1.5,
                 clahe: bool = True, clahe_clip_limit: float = 2.0,
                 video: bool = False):
        assert max_diff >= 0 and isinstance(max_diff, (int, float))
        assert min_diff >= 0 and isinstance(min_diff, (int, float))
        assert iteration > 0 and isinstance(iteration, int)
        assert gamma > 0 and isinstance(gamma, (int, float))
        assert isinstance(clahe, bool)
        assert isinstance(video, bool)

        self.max_diff = max_diff
        self.min_diff = min_diff
        self.iteration = iteration
        self.gamma = gamma
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.video = video
        self.prev_min_field = None
        self.prev_max_field = None

    def __call__(self, input: Union[str, np.ndarray]) -> np.ndarray:
        """Process an image or a path to an image. Expects 2D (grayscale) thermal."""
        if isinstance(input, str):
            image = cv2.imread(input, -1)
            if image is None:
                raise ValueError(f"Unable to read image from path: {input}")
        elif isinstance(input, np.ndarray):
            image = input
        else:
            raise TypeError("Input should be a file path or an numpy.ndarray.")

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = np.clip(image.astype(np.float64), 0, 65535)
            lo, hi = np.percentile(image, [2, 98])
            if hi > lo:
                image = np.clip((image - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        # 유효 픽셀 범위가 좁을 때(예: 90~99): 먼저 0~255로 스트레칭해 대비 확보
        mn, mx = image.min(), image.max()
        if mx > mn and (mx - mn) < 100:
            image = np.clip((image.astype(np.float64) - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)

        min_grid = gridwise_min(image, (8, 8))
        max_grid = gridwise_max(image, (8, 8))

        max_grid = local_extrema_suppression(
            max_grid, local_distance=2, diff_threshold=self.max_diff, extrema="max"
        )
        max_grid = local_extrema_suppression(
            max_grid, local_distance=2, diff_threshold=self.min_diff, extrema="min"
        )

        for _ in range(self.iteration):
            min_grid = message_passing(min_grid, direction="decrease").astype(np.float64)
            max_grid = message_passing(max_grid, direction="increase").astype(np.float64)

        min_field = cv2.resize(min_grid, dsize=(image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        max_field = cv2.resize(max_grid, dsize=(image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        if self.video and self.prev_min_field is not None:
            min_field = 0.1 * min_field + 0.9 * self.prev_min_field
            max_field = 0.1 * max_field + 0.9 * self.prev_max_field

        self.prev_min_field = min_field
        self.prev_max_field = max_field

        # rescaled = rescale_image_with_fields(image, min_field, max_field)
        rescaled = rescale_image_with_fields(image, min_field, max_field, min_range=2)
        

        if self.gamma > 0:
            rescaled = (255 * np.power(rescaled.astype(np.float64) / 255, self.gamma)).astype(np.uint8)

        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
            rescaled = clahe.apply(rescaled)

        return rescaled
