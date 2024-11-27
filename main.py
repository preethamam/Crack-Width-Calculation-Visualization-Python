import time
import sys

sys.dont_write_bytecode = True

from crackwidth import CrackAnalysis
from utils import ImageUtils
from visualize import CrackVisualization


def main():
    # Image path and filename
    image_path = "images/crack.bmp"

    # Perform crack width analysis
    analysis = CrackAnalysis(
        image_path,
        method="zhang",
        pixel_scale=1,
        mov_window_size=5,
        skel_orient_block_size=5,
        print_results=True,
    )

    # Perform skeletonization
    binary_crack, binary_skel = analysis.perform_skeletonization()

    # Save the binary skeleton image
    ImageUtils.save_image(binary_skel, "images/binary_skel.png")

    # Calculate orientations
    (
        orientations,
        onormal90,
        onormal270,
        onr_orient,
        onc_orient,
        onr90,
        onc90,
        onr270,
        onc270,
    ) = analysis.calculate_orientations()

    # Extract crack width
    xybresenham, mycell, row, col = analysis.extract_crack_width()

    # Calculate crack width using Bresenham's algorithm
    crack_width_bresenham, bresenham_cell = analysis.calculate_crack_width_bresenham(
        xybresenham
    )

    # Calculate crack metrics
    (
        crack_width_scaled,
        crack_length_scaled,
        min_crack_width,
        max_crack_width,
        average_crack_width,
        std_crack_width,
        rms_crack_width,
    ) = analysis.crack_metrics(crack_width_bresenham, bresenham_cell)

    # Visualize the crack width
    crackviz = CrackVisualization(
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
    )
    crackviz.binary_and_skeleton_plot()
    crackviz.crack_width_histogram()
    crackviz.crack_width_pdf()
    crackviz.crack_width_cdf()
    crackviz.crack_centerline()
    crackviz.crack_distance_map()
    crackviz.crack_normals()
    crackviz.crack_width_variation()
    crackviz.crack_width_variation_centerline()
    crackviz.crack_orientation_variation()
    crackviz.single_crack_width_visualization()
    crackviz.dynamic_crackline_movie()


if __name__ == "__main__":
    # Get the start time
    st1 = time.time()

    main()

    # Get the end time
    et1 = time.time()

    # Print the execution time
    elapsed_time = et1 - st1
    print(
        f"Algorithm Execution Time (Crack width analysis and visualization): {elapsed_time:.4f} seconds."
    )
