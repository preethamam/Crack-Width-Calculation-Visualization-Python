import numpy as np
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops
import math

class CrackWidthUtils:
    """
    A utility class for crack width calculations.
    """
    @staticmethod
    def bresenham(x1, y1, x2, y2):
        """
        Python implementation of the optimized Bresenham line algorithm.
        Parameters:
            x1, y1: Start position (integer coordinates)
            x2, y2: End position (integer coordinates)

        Returns:
            x, y: Numpy arrays of x and y coordinates for the line
        """
        # Round the inputs to ensure integer coordinates
        x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
        
        # Calculate differences
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steep = dy > dx

        # Swap coordinates if the line is steep
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            dx, dy = dy, dx

        # Determine the direction of the line
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        # Core algorithm
        err = dx // 2
        x, y = [], []

        while x1 != x2 + sx:  # Ensure we include the endpoint
            if steep:
                x.append(y1)  # Swap back for steep lines
                y.append(x1)
            else:
                x.append(x1)
                y.append(y1)

            err -= dy
            if err < 0:
                y1 += sy
                err += dx
            x1 += sx

        return np.array(x), np.array(y)
    
    @staticmethod
    def crackwidthlocation(x, y, angle, BW):
        """
        Determines the crack width location given a starting point, angle, and binary image.

        Args:
            x (int): Starting x-coordinate.
            y (int): Starting y-coordinate.
            angle (float): Angle in degrees.
            BW (np.ndarray): Binary image.

        Returns:
            dict: Dictionary with 'x' and 'y' keys containing lists of coordinates.
        """
        # Variables initialization
        line_length = 1
        yzeroflag, xzeroflag = 0, 0
        ymaxflag, xmaxflag = 0, 0
        xnew_array, ynew_array = [], []

        # Loop through pixels
        while True:
            if angle != 180 and angle != 360:
                xnew_width1 = int(np.floor(x + line_length * -np.cos(np.radians(angle))))
                ynew_width1 = int(np.floor(y + line_length * np.sin(np.radians(angle))))
            else:
                if angle == 180:
                    xnew_width1 = x - line_length
                    ynew_width1 = y
                elif angle == 360:
                    xnew_width1 = x + line_length
                    ynew_width1 = y

            if ynew_width1 <= 0:
                ynew_width1 = 0
                yzeroflag = 1
            if xnew_width1 <= 0:
                xnew_width1 = 0
                xzeroflag = 1
            if ynew_width1 >= BW.shape[0]:
                ynew_width1 = BW.shape[0] - 1
                ymaxflag = 1
            if xnew_width1 >= BW.shape[1]:
                xnew_width1 = BW.shape[1] - 1
                xmaxflag = 1

            xnew_array.append(xnew_width1)
            ynew_array.append(ynew_width1)

            if (BW[ynew_width1, xnew_width1] == 0 or 
                yzeroflag == 1 or xzeroflag == 1 or 
                ymaxflag == 1 or xmaxflag == 1 or 
                line_length > max(BW.shape)):
                break

            line_length += 1

        return {'x': xnew_array, 'y': ynew_array}

    @staticmethod
    def skeletonorientation(skel, blksz=None):
        """
        Calculate the local orientation of a skeleton.

        Args:
            skel (np.ndarray): 2D binary skeleton image.
            blksz (int or list, optional): Size of block to look around for local orientation.

        Returns:
            np.ndarray: Image of the same size as `skel` with orientations.
        """
        assert skel.dtype == bool, "skel should be a boolean array."
        assert skel.ndim == 2, "skel should be a 2D matrix."
        
        if blksz is None:
            blksz = [5, 5]
        else:
            if np.isscalar(blksz):
                blksz = [blksz, blksz]
            assert len(blksz) == 2, "blksz should be scalar or a 1x2 vector."
            assert all(x >= 3 for x in blksz), "blksz elements must be >= 3."
            assert all(x % 2 == 1 for x in blksz), "blksz elements must be odd."

        sz = skel.shape

        # Find the skeleton pixels' indices
        row, col = np.where(skel)
        npts = len(row)

        # Pad the array
        pad_amount = [x // 2 for x in blksz]
        skel_pad = np.pad(skel, ((pad_amount[0], pad_amount[0]), 
                                (pad_amount[1], pad_amount[1])), 
                        mode='constant', constant_values=0)

        # Preallocate orientations
        orientations = np.zeros(sz)

        # Offset values
        row_high = row + blksz[0] - 1
        col_high = col + blksz[1] - 1
        center = (pad_amount[0], pad_amount[1]) 

        s = generate_binary_structure(2,2)
        
        # Process each skeleton pixel
        for i in range(npts):
            # Extract small block
            block = skel_pad[row[i]:row_high[i] + 1, col[i]:col_high[i] + 1]

            # Label connected components
            labeled_block, _ = label(block, structure=s)
            center_label = labeled_block == labeled_block[center[0], center[1]]

            # Calculate orientation using regionprops
            rp = regionprops(center_label.astype(int))
            if rp:
                orientations[row[i], col[i]] = math.degrees(rp[0].orientation) + 90

        return orientations