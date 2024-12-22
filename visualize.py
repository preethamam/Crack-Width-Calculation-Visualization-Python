import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from scipy.stats import gaussian_kde
from utils import ImageManipulation


class CrackVisualization:
    """
    A class to visualize crack width and related properties in binary images.

    Attributes:
        binary_crack (np.ndarray): Binary image of the crack.
        binary_skel (np.ndarray): Binary image of the crack skeleton.
        orientations (np.ndarray): Array of crack orientations.
        onormal90 (np.ndarray): Array of normal orientations at 90 degrees.
        onormal270 (np.ndarray): Array of normal orientations at 270 degrees.
        onc_orient (np.ndarray): Array of crack orientations (column).
        onr_orient (np.ndarray): Array of crack orientations (row).
        onc90 (np.ndarray): Array of normal orientations at 90 degrees (column).
        onr90 (np.ndarray): Array of normal orientations at 90 degrees (row).
        onc270 (np.ndarray): Array of normal orientations at 270 degrees (column).
        onr270 (np.ndarray): Array of normal orientations at 270 degrees (row).
        bresenham_cell (list): List of Bresenham line cells.
        crack_width_scaled (np.ndarray): Array of scaled crack widths.
        row (np.ndarray): Array of row indices for crack centerline.
        col (np.ndarray): Array of column indices for crack centerline.
        crack_width_black_background (bool): Flag to use black background for crack width visualization.
        crack_index (int): Index of the crack to visualize.

    Methods:
        binary_and_skeleton_plot():
            Plots the binary image and the crack skeleton.
        crack_width_histogram():
            Displays a histogram of the crack thickness.
        crack_width_pdf():
            Displays the Probability Density Function (PDF) of the crack thickness.
        crack_width_cdf():
            Displays the Cumulative Distribution Function (CDF) of the crack thickness.
        crack_centerline():
            Displays the crack centerline on the binary image.
        crack_distance_map():
            Displays the crack distance map.
        crack_normals():
            Overlays the crack normals on the binary image.
        crack_width_variation():
            Visualizes the variation in crack width.
        crack_width_variation_centerline():
            Visualizes the variation in crack width along the centerline.
        crack_orientation_variation():
            Visualizes the variation in crack orientation along the centerline.
        single_crack_width_visualization():
            Visualizes the rasterization of a single crack width.
        dynamic_crackline_movie():
            Creates a dynamic movie of the crack width lines.
    """

    def __init__(
        self,
        binary_crack,
        binary_skel,
        orientations,
        onormal90,
        onormal270,
        onc_orient,
        onr_orient,
        onc90,
        onr90,
        onc270,
        onr270,
        bresenham_cell,
        crack_width_scaled,
        row,
        col,
        crack_width_black_background=False,
        crack_index=10
    ):
        """
        Initializes the CrackVisualization class.

        Args:
            binary_crack (np.ndarray): Binary image of the crack.
            binary_skel (np.ndarray): Binary image of the crack skeleton.
            orientations (np.ndarray): Array of crack orientations.
            onormal90 (np.ndarray): Array of normal orientations at 90 degrees.
            onormal270 (np.ndarray): Array of normal orientations at 270 degrees.
            onc_orient (np.ndarray): Array of crack orientations (column).
            onr_orient (np.ndarray): Array of crack orientations (row).
            onc90 (np.ndarray): Array of normal orientations at 90 degrees (column).
            onr90 (np.ndarray): Array of normal orientations at 90 degrees (row).
            onc270 (np.ndarray): Array of normal orientations at 270 degrees (column).
            onr270 (np.ndarray): Array of normal orientations at 270 degrees (row).
            bresenham_cell (list): List of Bresenham line cells.
            crack_width_scaled (np.ndarray): Array of scaled crack widths.
            row (np.ndarray): Array of row indices for crack centerline.
            col (np.ndarray): Array of column indices for crack centerline.
            crack_width_black_background (bool): Flag to use black background for crack width visualization.
            crack_index (int): Index of the crack to visualize.
        """
        self.binary_crack = binary_crack
        self.binary_skel = binary_skel
        self.orientations = orientations
        self.onormal90 = onormal90
        self.onormal270 = onormal270
        self.onc_orient = onc_orient
        self.onr_orient = onr_orient
        self.onc90 = onc90
        self.onr90 = onr90
        self.onc270 = onc270
        self.onr270 = onr270
        self.bresenham_cell = bresenham_cell
        self.crack_width_scaled = crack_width_scaled
        self.row = row
        self.col = col
        self.crack_width_black_background = crack_width_black_background
        self.crack_index = crack_index  # Example index

        # Create a modified 'jet' colormap with zero values set to black
        jet_cmap = plt.cm.jet
        new_colors = jet_cmap(np.linspace(0, 1, 256))
        new_colors[0] = [0, 0, 0, 1]  # Set the first color (zero values) to black
        self.custom_cmap = ListedColormap(new_colors)

        # Create the crack width factor image
        self.bw_width_factor = np.zeros_like(self.binary_crack, dtype=float)
        for m in range(len(self.bresenham_cell)):
            xnew_array = self.bresenham_cell[m][0]
            ynew_array = self.bresenham_cell[m][1]
            for n in range(len(xnew_array)):
                self.bw_width_factor[ynew_array[n], xnew_array[n]] = (
                    self.crack_width_scaled[m]
                )

    def binary_and_skeleton_plot(self):
        """
        Plots the binary image and the crack skeleton.
        """
        fig, axes = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 8)
        )

        ax = axes.ravel()

        ax[0].imshow(self.binary_crack, cmap=plt.cm.gray)
        ax[0].set_title("Binary image")
        ax[0].axis("off")
        
        ax[1].imshow(self.binary_skel, cmap=plt.cm.gray)
        ax[1].set_title("Crack Skeleton")        
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

    def crack_width_histogram(self):
        """
        Displays a histogram of the crack thickness.
        """
        plt.figure()
        plt.hist(self.crack_width_scaled, bins=30)
        plt.title("Histogram of the crack thickness")
        plt.xlabel("Thickness")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    def crack_width_pdf(self):
        """
        Displays the Probability Density Function (PDF) of the crack thickness.
        """
        kde = gaussian_kde(self.crack_width_scaled, bw_method="silverman")
        xi = np.linspace(0, max(self.crack_width_scaled), 100)
        f = kde(xi)
        plt.figure()
        plt.plot(xi, f)
        plt.title("PDF of the crack thickness")
        plt.xlabel("Thickness")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    def crack_width_cdf(self):
        """
        Displays the Cumulative Distribution Function (CDF) of the crack thickness.
        """
        plt.figure()
        sorted_data = np.sort(self.crack_width_scaled)
        cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, cdf)
        plt.title("CDF of the crack thickness")
        plt.xlabel("Thickness")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    def crack_centerline(self):
        """
        Displays the crack centerline on the binary image.
        """
        plt.figure()
        plt.imshow(self.binary_crack, cmap="gray", interpolation="none")
        plt.scatter(self.col, self.row, c="r", s=1)
        plt.title("Crack centerline")
        plt.axis("tight")
        plt.show()

    def crack_distance_map(self):
        """
        Displays the crack distance map.
        """
        yy, xx = np.arange(self.binary_crack.shape[0]), np.arange(
            self.binary_crack.shape[1]
        )
        X, Y = np.meshgrid(xx, yy)
        crackDistmapValues = self.binary_crack * (X + Y)

        plt.figure()
        img = plt.imshow(
            crackDistmapValues, cmap=self.custom_cmap, interpolation="none"
        )
        plt.axis("equal")
        plt.axis("tight")
        plt.axis("off")
        cbar = plt.colorbar(img)
        cbar.set_label("Crack Distance Measure")
        plt.scatter(self.col, self.row, color=[0.5, 0.5, 0.5], s=1)
        plt.show()

    def crack_normals(self):
        """
        Overlays the crack normals on the binary image.
        """
        plt.figure()
        plt.imshow(self.binary_crack, cmap="gray")
        plt.scatter(self.col, self.row, c="g", s=1)
        plt.quiver(
            self.col,
            self.row,
            [self.onc90[i][j] for i, j in zip(self.row, self.col)],
            [-self.onr90[i][j] for i, j in zip(self.row, self.col)],
            color="r",
            scale=90,
        )
        plt.quiver(
            self.col,
            self.row,
            [self.onc270[i][j] for i, j in zip(self.row, self.col)],
            [-self.onr270[i][j] for i, j in zip(self.row, self.col)],
            color="b",
            scale=90,
        )
        plt.quiver(
            self.col,
            self.row,
            [self.onc_orient[i][j] for i, j in zip(self.row, self.col)],
            [-self.onr_orient[i][j] for i, j in zip(self.row, self.col)],
            color="gray",
            scale=120,
        )
        plt.title("Crack normals visualization")
        plt.axis("tight")
        plt.axis("off")
        plt.show()

    def crack_width_variation(self):
        """
        Visualizes the variation in crack width.
        """
        if self.crack_width_black_background:
            plt.figure()
            plt.imshow(
                self.bw_width_factor, cmap=self.custom_cmap, interpolation="none"
            )
            plt.colorbar(label="Crack Width")
            plt.title("Crack width variation visualization")
            plt.axis("equal")
            plt.axis("off")
            plt.show()
        else:
            binary_crack_duplicate = self.binary_crack.copy()
            bw_width_factor = self.bw_width_factor * (
                self.bw_width_factor > 0
            )

            jet_cmap = plt.cm.jet(np.linspace(0, 1, 255))[:, :3]
            cmap = np.vstack(([0, 0, 0], jet_cmap))

            Norm_BWWF = bw_width_factor.copy()
            min_val = np.min(Norm_BWWF[Norm_BWWF > 0])
            max_val = np.max(Norm_BWWF)

            Norm_BWWF = (
                (Norm_BWWF - min_val) / (max_val - min_val) * (len(cmap) - 1)
            ).astype(int)
            Norm_BWWF[bw_width_factor == 0] = 0

            imgsc_im = cmap[Norm_BWWF]

            MaskBW = np.repeat(
                (binary_crack_duplicate > 0) & (bw_width_factor == 0), 3
            ).reshape(self.binary_crack.shape + (3,))
            imgsc_im[MaskBW] = 1

            plt.figure()
            plt.imshow(imgsc_im)
            plt.axis("equal")
            plt.axis("off")
            plt.title("Crack width variation visualization")

            sm = plt.cm.ScalarMappable(
                cmap="jet", norm=Normalize(vmin=min_val, vmax=max_val)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label("Crack Width")
            plt.show()

    def crack_width_variation_centerline(self):
        """
        Visualizes the variation in crack width along the centerline.
        """
        bw_width_factorCL = self.binary_skel * self.bw_width_factor
        plt.figure()
        plt.imshow(bw_width_factorCL, cmap=self.custom_cmap, interpolation="none")
        plt.colorbar(label="Crack Width")
        plt.title("Crack width variation visualization along center line")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def crack_orientation_variation(self):
        """
        Visualizes the variation in crack orientation along the centerline.
        """
        CLOTangential = self.binary_skel * np.abs(self.orientations)
        CLOnormal = self.onormal90

        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        ax = axes.ravel()

        im1 = ax[0].imshow(CLOTangential, cmap=self.custom_cmap, interpolation="none")
        ax[0].set_title("Crack tangential angle variation along center line")
        ax[0].axis("off")
        fig.colorbar(im1, label="Tangential Angle (°)", ax=ax[0], shrink=0.5)

        im2 = ax[1].imshow(CLOnormal, cmap=self.custom_cmap, interpolation="none")
        ax[1].set_title("Crack normal angle variation along center line")
        ax[1].axis("off")
        fig.colorbar(im2, label="Normal Angle (°)", ax=ax[1], shrink=0.5)
        plt.tight_layout()
        plt.show()

    def single_crack_width_visualization(self):
        """
        Visualizes the rasterization of a single crack width.
        """
        single_crack_width = np.zeros_like(self.binary_crack, dtype=bool)

        crackCL = ImageManipulation.imoverlay(
            self.binary_crack, self.binary_skel, [1, 0, 0]
        )

        x_bresenham = self.bresenham_cell[self.crack_index][0]
        y_bresenham = self.bresenham_cell[self.crack_index][1]

        for j in range(len(x_bresenham)):
            single_crack_width[y_bresenham[j], x_bresenham[j]] = True

        crackWidthStrand = ImageManipulation.imoverlay(
            crackCL, single_crack_width, [0, 0, 1]
        )

        plt.figure()
        plt.imshow(crackWidthStrand, interpolation="none")
        plt.title("Rasterization visualization of a single crack width")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def dynamic_crackline_movie(self):
        """
        Creates a dynamic movie of the crack width lines.
        """
        greenCL = ImageManipulation.imoverlay(self.binary_crack, self.binary_skel, [0, 1, 0])

        plt.figure()
        plt.imshow(greenCL)
        plt.title('Crack width lines movie')
        plt.axis('off')
        plt.show(block=False)

        for m in range(len(self.bresenham_cell)):
            xnew_array, ynew_array = self.bresenham_cell[m]
            plt.plot(xnew_array, ynew_array)
            plt.draw()
            plt.pause(0.000001)

        plt.show()