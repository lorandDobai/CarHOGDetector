import math
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numba import jit, boolean, float64, uint16, void, int8, float32, int16, autojit, uint8,bool_
from skimage import img_as_float
from skimage.color import rgb2gray, gray2rgb
import ujson as json
import skimage.io
import warnings


def display(images, colormap = plt.cm.gray):
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(1, len(images), figsize=(8, 4), sharex=True, sharey=True)
    axes = [axes] if len(images) == 1 else axes
    for img,ax in zip(images,axes):
        ax.imshow(img, cmap=colormap)
    plt.show()


def get_bounding_box_angles(boxes):
    ret = []
    for box in boxes:
        diff = box[2] - box[1]
        ret.append(np.degrees(np.arctan2(diff[1], diff[0])))
    return np.array(ret)

def load_own_json_annotation(path):
    with open(path) as fin:
        data = json.loads(fin.read())
    ret_data = []
    for element in data:
        coords = np.array(element['coords']).astype(dtype=np.float)
        angle = element['angle']
        ret_coord = np.empty_like(coords)
        rotate_coords(coords, angle, element['translateX'], element['translateY'], ret_coord)
        ret_data.append(ret_coord[:].astype(dtype=np.float64))
    return ret_data


def load_multiclass_json_annotation(path):
    with open(path) as fin:
        data = json.loads(fin.read())
    data['class'] = np.array(data['class_'])
    return data

@jit(nopython=True)
def rotate_coords(coords,angle,translateX,translateY,return_arr):
    xg = np.sum(coords[:, 0])/4
    yg = np.sum(coords[:, 1])/4
    for i in range(len(coords)):
        x,y = coords[i]
        x += translateX
        y += translateY
        x -= xg
        y -= yg
        xr = x*math.cos(angle) - y * math.sin(angle)
        yr = x*math.sin(angle) + y * math.cos(angle)
        xr += xg
        yr += yg
        return_arr[i,0], return_arr[i,1] = xr, yr


def load_oxford_matlab_annotation(path):
    ret = []
    # print(scipyio.loadmat(path)['boxes'])
    for arr in (scipy.io.loadmat(path)['boxes'][0]):

        ret.append([(box[0][1], box[0][0]) for box in arr[0][0] if box.size==2])
    return np.array(ret)


def plot_bounding_box(img, data, color='r',axis='on'):

    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    plt.axis(axis)
    for box in data:
        ax1.plot((box[0][0], box[1][0]), (box[0][1], box[1][1]), color=color)
        ax1.plot((box[1][0], box[2][0]), (box[1][1], box[2][1]), color=color)
        ax1.plot((box[2][0], box[3][0]), (box[2][1], box[3][1]), color=color)
        ax1.plot((box[3][0], box[0][0]), (box[3][1], box[0][1]), color=color)

    ax1.imshow(img, cmap=plt.cm.gray)
    plt.show()


@jit(void(float64[:, :], uint16, float64[:], float64), nopython=True)
def rotate_scale_bounding_box(box, angle, img_center, scale_factor):
    """
    Rotate and scale box coordinates
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


@jit(void(float64[:, :, :],uint16[:, :],uint8,uint8,uint8, float64[:,:]), nopython=True)
def extract_hog_features(hog_blocks,window_positions,cell_size,blocksize,n_bins,output_feature_array):

    features_per_block = n_bins*blocksize**2
    for c in range(window_positions.shape[0]):
        y,ye,x,xe = window_positions[c]
        counter = 0
        for i in range(y, ye):
            for j in range(x, xe):
                output_feature_array[c][counter:counter+features_per_block] = hog_blocks[i, j]
                counter+=features_per_block

        # min_value = np.min(output_feature_array[c])
        # output_feature_array[c] = (output_feature_array[c]-min_value)/(np.max(output_feature_array[c])-min_value)
        window_positions[c][1] = window_positions[c][1] - 1 + blocksize
        window_positions[c][3] = window_positions[c][3] - 1 + blocksize
        window_positions[c] *= cell_size





def fast_custom_hog_blocks(input_img, cell_size=6, n_bins=9, blocksize=3):

    hog = custom_hog(input_img, cell_size, n_bins)
    hog_height, hog_width, bins = hog.shape
    hog_blocks = np.empty(((hog_height+1-blocksize), (hog_width+1-blocksize), n_bins * blocksize**2), dtype=hog.dtype)
    normalize_hog_blocks(hog, hog_blocks,blocksize)
    return hog_blocks


@jit(void(float64[:]), nopython=True)
def fast_normalize(numba_arr):
    geom_sum = 0
    for i in range(len(numba_arr)):
        geom_sum += numba_arr[i] ** 2
    norm_factor = math.sqrt(math.sqrt(geom_sum) ** 2 + 0.005 ** 2)
    for i in range(len(numba_arr)):
        numba_arr[i] /= norm_factor


@jit(void(float64[:, :, :], float64[:, :, :], uint8),nopython=True)
def normalize_hog_blocks(hog, out_matrix, blocksize):
    for y in range(out_matrix.shape[0]):
        for x in range(out_matrix.shape[1]):
            current_block = hog[y:y+blocksize, x:x+blocksize].copy().ravel()
            fast_normalize(current_block)
            out_matrix[y, x] = current_block



def custom_hog(img, cell_size=6, n_bins=9):
    """
    **SUMMARY**
    Get HOG(Histogram of Oriented Gradients) features from the image.


    **PARAMETERS**
    * *img*    - Numpy array instance
    * *cell_size* - the number of divisions(cells).
    * *n_bins* - the number of orientation bins.

    **RETURNS**
    Returns the HOG vector in a numpy array
    """
    height, width,  = img.shape[:2]
    # number of cells
    h_divs, w_divs = height // cell_size, width // cell_size

    # Size of HOG vector
    n_HOG = h_divs * w_divs * n_bins
    # Apply filter on image to find magnitude of gradient and angles.
    magnit = np.zeros((height,width),dtype=np.float64)
    angles = np.zeros((height,width),dtype=np.float64)
    rgb_numba_gradient(img, magnit, angles)

    bin_range = (2 * math.pi) / n_bins
    bins = (angles % (2 * math.pi) / bin_range).astype(np.uint16)
    y, x = np.mgrid[:height, :width]
    x = x * w_divs // width
    y = y * h_divs // height

    labels = (y * w_divs + x) * n_bins + bins
    index = np.arange(n_HOG)
    HOG = ndimage.measurements.sum(magnit, labels, index)
    return HOG.reshape(h_divs, w_divs, n_bins)


@jit(boolean(float64[:, :, :], uint16, uint16, uint16, uint16), nopython=True)
def is_empty_window(hog_matrix, y, ye, x, xe):
    return hog_matrix[y:ye, x:xe].max() == 0


@jit(boolean(float64[:, :], uint16, uint16, uint16, uint16), nopython=True)
def is_empty_window_img(hog_matrix, y, ye, x, xe):
    return hog_matrix[y:ye, x:xe].max() == 0


@jit(void(uint16, uint16, uint16, uint16, uint16[:, :]), nopython=True)
def get_window_positions(hog_height, hog_width, window_height, window_width, out_arr):
    count = 0
    for y in range(hog_height - window_height + 1):
        for x in range(hog_width - window_width + 1):
            out_arr[count][0] = y
            out_arr[count][1] = y + window_height
            out_arr[count][2] = x
            out_arr[count][3] = x + window_width
            count += 1


@jit(void(float64[:, :, :], int8, int8, float64[:]), nopython=True)
def calculate_feature_array(window_tiles, block_size, n_bins, out_arr):
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
    height, width = img.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sx = img[y, x + 1] - img[y, x - 1]
            sy = img[y + 1, x] - img[y - 1, x]
            out_mag[y, x] = math.sqrt(sx ** 2 + sy ** 2)
            out_angle[y, x] = math.atan2(sy, sx)


@jit(float64(float32[:, :], uint16[:, :]), nopython=True)
def overlap(annot_box, sliding_window_coord):
    """
        Calculate overlap between the 2 boxes
        Rectangle extremities (box[2],box[0]) for oxford dataset annotations
        Formula for intersection is SI= Max(0, Min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))
    """
    a1, a2 = sliding_window_coord[0], sliding_window_coord[1]
    b1, b2 = annot_box[0], annot_box[1]

    S1 = (a2[0] - a1[0]) * (a2[1] - a1[1])
    S2 = (b1[0] - b2[0]) * (b1[1] - b2[1])
    SI = max(0, min(a2[0], b2[0]) - max(a1[0], b1[0])) * max(0, min(a2[1], b2[1]) - max(a1[1], b1[1]))
    S = S1 + S2 - SI
    return SI / S


@jit(float64(float32[:, :], uint16[:, :]), nopython=True)
def overlap_over_small_box_area(annot_box,sliding_window_coord):
    a1, a2 = sliding_window_coord[0], sliding_window_coord[1]
    b1, b2 = annot_box[0], annot_box[1]
    S1 = (a2[0] - a1[0]) * (a2[1] - a1[1])
    S2 = (b1[0] - b2[0]) * (b1[1] - b2[1])
    SI = max(0, min(a2[0], b2[0]) - max(a1[0], b1[0])) * max(0, min(a2[1], b2[1]) - max(a1[1], b1[1]))
    S = S1 + S2 - SI
    return SI / min(S1,S2)


@jit(void(float32[:, :]), nopython=True)
def inplace_logistic_sigmoid(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = 1/(1+math.exp(-arr[i, j]))



@jit(void(float64[:, :]), nopython=True)
def inplace_logistic_sigmoidf64(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = 1/(1+math.exp(-arr[i, j]))


def load_image(path):
    img = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img = img_as_float(skimage.io.imread(path))
        if len(img.shape)== 2:
            img = img_as_float(gray2rgb(img))
    return img

if __name__ == '__main__':
    #data = load_own_json_annotation('K:/Licenta/licenta2016/Data/MyDataSet1/annotations_json/time-person.json')
    #img = skimage.io.imread('Data/MyDataSet1/images/time-person.jpg')
    #plot_bounding_box(img,data)
    img = img_as_float(rgb2gray(skimage.io.imread("K:\\Licenta\\licenta2016\\Data\\MyDataSet2\\images\\talk-to-the-hand.png")))
    custom_hog(img)
    # out_mag = np.zeros((img.shape[0],img.shape[1]),dtype=np.float64)
    # out_angle = np.zeros((img.shape[0],img.shape[1]),dtype=np.float64)
    # rgb_numba_gradient(img,out_mag,out_angle)
    # skimage.io.imshow(out_mag)
    # skimage.io.show()
    #
    # img_gray = rgb2gray(img)
    # gray_numba_gradient(img_gray,out_mag,out_angle)
    # skimage.io.imshow(out_mag)
    # skimage.io.show()
    # import time
    # ht = time.time()
    # res = fast_custom_hog_blocks(img)
    # print(time.time()-ht)
    # ht = time.time()
    # fast_custom_hog_blocks(img)
    # print(time.time()-ht)