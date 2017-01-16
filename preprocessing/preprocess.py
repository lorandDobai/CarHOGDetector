import math
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, boolean, float64, uint16, void, int8, float32, int16,int32, autojit, uint8,bool_



def display(images, colormap = plt.cm.gray):
    """
    Plot images and displays them.

    :param images:  list of images to be displayed
    :type images: List[np.ndarray]
    :param colormap: (Optional) colormap of the plot
    :return: None
    """
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(1, len(images), figsize=(8, 4), sharex=True, sharey=True)
    axes = [axes] if len(images) == 1 else axes
    for img,ax in zip(images,axes):
        ax.imshow(img, cmap=colormap)
    plt.show()



def plot_bounding_box(img, data, color='r',axis='on'):
    """
    Plot and display bounding boxes on given image

    :param img: image on which to plot the bounding boxes
    :type img: np.ndarray
    :param data: list of the bounding boxes of format [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :param color: (Optional) color of the plotted box
    :type color: string
    :param axis: show axis on plot. 'on' or 'off'
    :type axis: string
    :return: None
    """
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    plt.axis(axis)
    for box in data:
        ax1.plot((box[0][0], box[1][0]), (box[0][1], box[1][1]),linewidth=2, color=color)
        ax1.plot((box[1][0], box[2][0]), (box[1][1], box[2][1]),linewidth=2, color=color)
        ax1.plot((box[2][0], box[3][0]), (box[2][1], box[3][1]),linewidth=2, color=color)
        ax1.plot((box[3][0], box[0][0]), (box[3][1], box[0][1]),linewidth=2, color=color)

    ax1.imshow(img, cmap=plt.cm.gray)
    plt.show()


@jit(void(float64[:, :], uint16, float64[:], float64), nopython=True)
def rotate_scale_bounding_box(box, angle, img_center, scale_factor):
    """
    Rotate and scale box coordinates

    :param box: Box(rectangle) coordinates of format [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :type box: List[int]
    :param angle:
    :param img_center:
    :param scale_factor:
    :return: None
    """
    angle = math.radians(angle)
    for i in range(len(box)):
        # Translate to origin
        coord_T = box[i] - img_center
        # Rotate
        coord_Rx = coord_T[0] * math.cos(angle) - coord_T[1] * math.sin(angle)
        coord_Ry = coord_T[0] * math.sin(angle) + coord_T[1] * math.cos(angle)
        # Translate back
        box[i] = (np.array((coord_Rx, coord_Ry)) + img_center) * scale_factor




def custom_hog(img, cell_size=6, n_bins=9):
    """
    Get HOG(Histogram of Oriented Gradients) features from the image.

    :param img: input greyscale image
    :type img: np.ndarray
    :param cell_size: dimension of a spatial division
    :type cell_size: int
    :param n_bins: number of orientation divisions
    :type n_bins: int
    :return: HOG cell matrix
    :rtype: np.ndarray
    """
    height, width,  = img.shape[:2]
    # number of cells on y-axis, respectively x-axis
    h_divs, w_divs = height // cell_size, width // cell_size
    # Size of HOG vector
    n_HOG = h_divs * w_divs * n_bins
    # Apply filter on image to find magnitude of gradient and angles.
    magnit = np.zeros((height,width), dtype=np.float64)
    angles = np.zeros((height,width), dtype=np.float64)
    gray_numba_gradient(img, magnit, angles)

    # Find the bin of each magnitude elemnt
    bin_range = (2 * math.pi) / n_bins
    bins = (angles % (2 * math.pi) / bin_range).astype(np.uint16)
    y, x = np.mgrid[:height, :width]
    x = x * w_divs // width
    y = y * h_divs // height
    labels = (y * w_divs + x) * n_bins + bins
    index = np.arange(n_HOG)

    # Calculate the histograms for each cell
    HOG = ndimage.measurements.sum(magnit, labels, index)
    return HOG.reshape(h_divs, w_divs, n_bins)


@jit(boolean(float64[:, :, :], uint16, uint16, uint16, uint16), nopython=True)
def is_empty_window(hog_matrix, y, ye, x, xe):
    """
    Check if sliding window of detector is over empty(black) region

    :param hog_matrix: HOG cells contained in the window
    :param y: top y coordinate
    :param ye: bottom y coordinate
    :param x: leftmost x coordinate
    :param xe: rightmost x coordinate
    :return: True if input is empty region else False
    """
    return hog_matrix[y:ye, x:xe].max() == 0



@jit(void(float64[:, :, :], int8, int8, float64[:]), nopython=True)
def hog_feature_arr(window_tiles, block_size, n_bins, out_arr):
    """
    Computes the normalized HOG feature array from raw HOG cells of current window.

    :param window_tiles: Non-normalized HOG cells of sliding window
    :param block_size: number of cells in a block
    :param n_bins: number of orientation bins
    :param out_arr: array in which output is stored (avoiding memory allocations)
    :return: None
    """
    h, w = window_tiles.shape[:2]
    step = 0
    features_per_block = n_bins * (block_size ** 2)

    for y in range(len(window_tiles)):
        if y + block_size <= h:
            for x in range(len(window_tiles[y])):
                if x + block_size <= w:
                    block = window_tiles[y:y + block_size, x:x + block_size]
                    numba_arr = block.copy().ravel()

                    geom_sum = 0
                    for i in range(len(numba_arr)):
                        geom_sum += numba_arr[i] ** 2

                    norm_factor = math.sqrt(math.sqrt(geom_sum) ** 2 + 0.005 ** 2)
                    for i in range(len(numba_arr)):
                        numba_arr[i] /= norm_factor

                    out_arr[step:step + features_per_block] = numba_arr
                    step += features_per_block


@jit(void(float64[:, :, :], float64[:, :], float64[:, :]), nopython=True)
def rgb_numba_gradient(img, out_mag, out_angle):
    """
    Computes magnitude & orientation grid for RGB images.

    :param img: input RGB image
    :param out_mag: magnitude grid
    :param out_angle: orientation grid
    :return:
    """
    height, width, _ = img.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sxr = img[y, x + 1, 0] - img[y, x - 1, 0]
            syr = img[y + 1, x, 0] - img[y - 1, x, 0]

            mag_value= math.sqrt(sxr**2 + syr**2)
            vals = sxr,syr

            sxg = img[y, x + 1, 1] - img[y, x - 1, 1]
            syg = img[y + 1, x, 1] - img[y - 1, x, 1]

            tmp = math.sqrt(sxg**2 + syg**2)
            if tmp > mag_value:
                mag_value = tmp
                vals = sxg, syg

            sxb = img[y, x + 1, 2] - img[y, x - 1, 2]
            syb = img[y + 1, x, 2] - img[y - 1, x, 2]

            tmp = math.sqrt(sxb**2 + syb**2)
            if tmp > mag_value:
                mag_value = tmp
                vals = sxb, syb

            out_mag[y, x] = mag_value
            out_angle[y, x] = math.atan2(vals[1], vals[0])


@jit(void(float64[:, :], float64[:, :], float64[:, :]), nopython=True)
def gray_numba_gradient(img, out_mag, out_angle):
    """
    Computes magnitude & orientation grid for Greyscale images.

    :param img: input RGB image
    :param out_mag: magnitude grid
    :param out_angle: orientation grid
    :return:
    """
    height, width = img.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sx = img[y, x + 1] - img[y, x - 1]
            sy = img[y + 1, x] - img[y - 1, x]
            out_mag[y, x] = math.sqrt(sx ** 2 + sy ** 2)
            out_angle[y, x] = math.atan2(sy, sx)


@jit(float64(int32[:, :], int32[:, :]), nopython=True)
def overlap(annot_box, sliding_window):
    """
    Returns overlap proportion between the 2 rectangles.
    Overlap is defined as Area of intersection divided by total area of the two rectangles
    Rectangle extremities ((XA1,YA1),(XA2,YA2)) are used.
    Formula for intersection is SI= Max(0, Min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))

    :param annot_box: First Rectangle
    :param sliding_window: Second Rectangle
    :return: Overlap Proportion
    """
    a1, a2 = sliding_window[0], sliding_window[1]
    b1, b2 = annot_box[0], annot_box[1]

    S1 = (a2[0] - a1[0]) * (a2[1] - a1[1])
    S2 = (b1[0] - b2[0]) * (b1[1] - b2[1])
    SI = max(0, min(a2[0], b2[0]) - max(a1[0], b1[0])) * max(0, min(a2[1], b2[1]) - max(a1[1], b1[1]))
    S = S1 + S2 - SI
    return SI / S

def get_hog_feature_array(img, feature_count, cell_size, block_size, n_bins):

    feature_arr = np.empty((feature_count,), dtype=np.float64)
    hog_feature_arr(custom_hog(img, cell_size=cell_size, n_bins = n_bins),\
                   block_size, n_bins, feature_arr)
    return feature_arr



