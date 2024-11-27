import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from scipy.stats import gaussian_kde
from utils import ImageManipulation


class CrackVisualization:

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
        # Plot the skeleton and the binary image
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
        # Display histogram
        plt.figure()
        plt.hist(self.crack_width_scaled, bins=30)
        plt.title("Histogram of the crack thickness")
        plt.xlabel("Thickness")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    def crack_width_pdf(self):
        # Display PDF
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
        # Display CDF
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
        # Display crack centerline
        plt.figure()
        plt.imshow(self.binary_crack, cmap="gray", interpolation="none")
        plt.scatter(self.col, self.row, c="r", s=1)
        plt.title("Crack centerline")
        plt.axis("tight")
        plt.show()

    def crack_distance_map(self):
        # Find distance map
        # Create meshgrid for X and Y
        yy, xx = np.arange(self.binary_crack.shape[0]), np.arange(
            self.binary_crack.shape[1]
        )
        X, Y = np.meshgrid(xx, yy)

        # Calculate crack distance map values
        crackDistmapValues = self.binary_crack * (X + Y)

        # Plot the distance map
        plt.figure()
        img = plt.imshow(
            crackDistmapValues, cmap=self.custom_cmap, interpolation="none"
        )
        plt.axis("equal")
        plt.axis("tight")
        plt.axis("off")

        # Add the colorbar and set its label
        cbar = plt.colorbar(img)
        cbar.set_label("Crack Distance Measure")

        # Overlay additional features
        plt.scatter(self.col, self.row, color=[0.5, 0.5, 0.5], s=1)
        plt.show()

    def crack_normals(self):
        # Overlay normals to verify
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
        # Plot the crack width factor
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
            # Black background with white crack width region
            binary_crack_duplicate = self.binary_crack.copy()
            bw_width_factor = self.bw_width_factor * (
                self.bw_width_factor > 0
            )  # Ensure non-negative values

            # Define the colormap: black and jet
            jet_cmap = plt.cm.jet(np.linspace(0, 1, 255))[:, :3]
            cmap = np.vstack(([0, 0, 0], jet_cmap))  # Add black as the first color

            # Normalize bw_width_factor to map to colormap indices
            Norm_BWWF = bw_width_factor.copy()
            min_val = np.min(Norm_BWWF[Norm_BWWF > 0])  # Minimum non-zero value
            max_val = np.max(Norm_BWWF)

            Norm_BWWF = (
                (Norm_BWWF - min_val) / (max_val - min_val) * (len(cmap) - 1)
            ).astype(int)
            Norm_BWWF[bw_width_factor == 0] = 0  # Ensure zero values map to black

            # Convert normalized values to RGB image
            imgsc_im = cmap[Norm_BWWF]

            # Create mask for pixels in BW4 but with zero width factor
            MaskBW = np.repeat(
                (binary_crack_duplicate > 0) & (bw_width_factor == 0), 3
            ).reshape(self.binary_crack.shape + (3,))
            imgsc_im[MaskBW] = 1  # Assign white to these pixels

            # Plot the resulting image
            plt.figure(figsize=(8, 6))
            plt.imshow(imgsc_im)
            plt.axis("equal")
            plt.axis("off")
            plt.title("Crack width variation visualization")

            # Add colorbar with jet colormap
            sm = plt.cm.ScalarMappable(
                cmap="jet", norm=Normalize(vmin=min_val, vmax=max_val)
            )
            sm.set_array([])  # Set the array explicitly
            cbar = plt.colorbar(sm, ax=plt.gca())  # Tie colorbar to the current Axes
            cbar.set_label("Crack Width")
            plt.show()

    def crack_width_variation_centerline(self):
        # Plot the crack width factor center line
        bw_width_factorCL = self.binary_skel * self.bw_width_factor
        plt.figure()
        plt.imshow(bw_width_factorCL, cmap=self.custom_cmap, interpolation="none")
        plt.colorbar(label="Crack Width")
        plt.title("Crack width variation visualization along center line")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def crack_orientation_variation(self):
        # Crack tangential and normal orientations
        CLOTangential = self.binary_skel * np.abs(self.orientations)
        CLOnormal = self.onormal90

        # Plot the skeleton and the binary image
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
        # Single crack width visualization
        single_crack_width = np.zeros_like(self.binary_crack, dtype=bool)

        # Overlay the first mask
        crackCL = ImageManipulation.imoverlay(
            self.binary_crack, self.binary_skel, [1, 0, 0]
        )

        # Extract the crack index
        x_bresenham = self.bresenham_cell[self.crack_index][0]
        y_bresenham = self.bresenham_cell[self.crack_index][1]

        # Set the crack width pixels to True
        for j in range(len(x_bresenham)):
            single_crack_width[y_bresenham[j], x_bresenham[j]] = True

        # Overlay the crack width visualization
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
        # Create overlay with binarySkeleton in green on top of binaryCrack
        greenCL = ImageManipulation.imoverlay(self.binary_crack, self.binary_skel, [0, 1, 0])

        plt.figure()
        plt.imshow(greenCL)
        plt.title('Crack width lines movie')
        plt.axis('off')
        plt.show(block=False)  # Show image without blocking further updates

        # Plot Bresenham lines on the overlay
        for m in range(len(self.bresenham_cell)):
            xnew_array, ynew_array = self.bresenham_cell[m]
            plt.plot(xnew_array, ynew_array)
            plt.draw()
            plt.pause(0.000001)  # Pause for visual effect

        plt.show()  # Show the final image
