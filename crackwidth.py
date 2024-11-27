from scipy.ndimage import uniform_filter1d
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize

import numpy as np

from crackutils import CrackWidthUtils


class CrackAnalysis:
    def __init__(
        self,
        image_path,
        method="zhang",
        pixel_scale=1,
        mov_window_size=5,
        skel_orient_block_size=5,
        print_results=True,
    ):
        self.image_path = image_path
        self.method = method
        self.pixel_scale = pixel_scale
        self.mov_window_size = mov_window_size
        self.skel_orient_block_size = skel_orient_block_size
        self.print_results = print_results
        self.binary_crack = imread(image_path, as_gray=True) > 0
        self.binary_skel = None
        self.orientations = None
        self.onormal90 = None
        self.onormal270 = None
        self.onr_orient = None
        self.onc_orient = None
        self.onr90 = None
        self.onc90 = None
        self.onr270 = None
        self.onc270 = None    

    def perform_skeletonization(self):
        if self.method == "zhang":
            self.binary_skel = skeletonize(self.binary_crack > 0)
        elif self.method == "lee":
            self.binary_skel = skeletonize(self.binary_crack > 0, method="lee")
        elif self.method == "medial_axis":
            self.binary_skel, _ = medial_axis(
                self.binary_crack > 0, return_distance=True
            )
        else:
            raise ValueError(
                f"Invalid skeletonization method: {self.method}. Choose 'lee' or 'medial_axis'."
            )
        return self.binary_crack, self.binary_skel
    
    def calculate_orientations(self):
        self.orientations = CrackWidthUtils.skeletonorientation(
            self.binary_skel, self.skel_orient_block_size
        )
        self.onormal90 = self.binary_skel * (self.orientations + 90)
        self.onormal270 = self.binary_skel * (self.orientations + 270)
        self.onr_orient = np.sin(np.radians(self.orientations))
        self.onc_orient = np.cos(np.radians(self.orientations))
        self.onr90 = np.sin(np.radians(self.onormal90))
        self.onc90 = np.cos(np.radians(self.onormal90))
        self.onr270 = np.sin(np.radians(self.onormal270))
        self.onc270 = np.cos(np.radians(self.onormal270))
        
        return self.orientations, self.onormal90, self.onormal270, self.onr_orient, self.onc_orient, self.onr90, self.onc90, self.onr270, self.onc270

    def extract_crack_width(self):
        row, col = np.where(self.binary_skel)
        angle_1 = [self.onormal90[i][j] for i, j in zip(row, col)]
        angle_2 = [self.onormal270[i][j] for i, j in zip(row, col)]
        mycell = [[None] * len(row) for _ in range(2)]
        xybresenham = np.zeros((len(angle_1), 4))

        for i in range(len(row)):
            mycell[0][i] = CrackWidthUtils.crackwidthlocation(
                col[i], row[i], angle_1[i], self.binary_crack
            )
            mycell[1][i] = CrackWidthUtils.crackwidthlocation(
                col[i], row[i], angle_2[i], self.binary_crack
            )
            xybresenham[i, :] = [
                mycell[0][i]["x"][-1],
                mycell[0][i]["y"][-1],
                mycell[1][i]["x"][-1],
                mycell[1][i]["y"][-1],
            ]

        return xybresenham, mycell, row, col

    def calculate_crack_width_bresenham(self, xybresenham):
        bresenham_cell = [[None, None] for _ in range(len(xybresenham))]
        crack_width_bresenham = np.zeros(len(xybresenham))

        for i in range(len(xybresenham)):
            x_bresenham, y_bresenham = CrackWidthUtils.bresenham(
                xybresenham[i, 0],
                xybresenham[i, 1],
                xybresenham[i, 2],
                xybresenham[i, 3],
            )
            bresenham_cell[i][0] = x_bresenham
            bresenham_cell[i][1] = y_bresenham
            crack_width_bresenham[i] = len(x_bresenham)

        return crack_width_bresenham, bresenham_cell

    def crack_metrics(self, crack_width_bresenham, bresenham_cell):
        crack_width_scaled = crack_width_bresenham * self.pixel_scale
        crack_length_scaled = len(np.flatnonzero(self.binary_skel)) * self.pixel_scale
        crack_width_scaled = uniform_filter1d(
            crack_width_scaled, size=self.mov_window_size
        )

        min_crack_width = np.min(crack_width_scaled)
        max_crack_width = np.max(crack_width_scaled)
        average_crack_width = np.mean(crack_width_scaled)
        std_crack_width = np.std(crack_width_scaled)
        rms_crack_width = np.sqrt(np.mean(crack_width_scaled**2))

        if self.print_results:
            print(f"Minimum Crack Width: {min_crack_width}")
            print(f"Maximum Crack Width: {max_crack_width}")
            print(f"Average Crack Width: {average_crack_width}")
            print(f"Standard Deviation of Crack Width: {std_crack_width}")
            print(f"RMS Crack Width: {rms_crack_width}")
            print(f"Crack Length: {crack_length_scaled}")

        return (
            crack_width_scaled,
            crack_length_scaled,
            min_crack_width,
            max_crack_width,
            average_crack_width,
            std_crack_width,
            rms_crack_width,
        )
