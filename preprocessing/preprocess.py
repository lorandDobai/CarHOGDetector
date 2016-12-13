import os
import ujson as json
import skimage.io
import skimage.transform
import numpy as np
import math
import matplotlib.pyplot as plt
import random
def get_bounding_box_angles(boxes):
    ret = []
    for box in boxes:
        diff = box[2] - box[1]
        ret.append(np.degrees(np.arctan2(diff[1], diff[0])))
    return np.array(ret)

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

def overlap(box1, box2):
    """
        Calculate overlap between the 2 boxes
        Rectangle extremities (box[2],box[0]) for oxford dataset annotations
        Formula for intersection is SI= Max(0, Min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))
    """
    a1, a2 = box1[0], box1[1]
    b1, b2 = box2[0], box2[1]

    S1 = (a2[0] - a1[0]) * (a2[1] - a1[1])
    S2 = (b1[0] - b2[0]) * (b1[1] - b2[1])
    SI = max(0, min(a2[0], b2[0]) - max(a1[0], b1[0])) * max(0, min(a2[1], b2[1]) - max(a1[1], b1[1]))
    S = S1 + S2 - SI
    return SI / S

def plot_bounding_box(img, data, color='r',axis='on'):

    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    plt.axis(axis)
    if type(color) is not list:
        color = [color]*len(data)
    for i,box in enumerate(data):
        ax1.plot((box[0][0], box[1][0]), (box[0][1], box[1][1]), color=color[i])
        ax1.plot((box[1][0], box[2][0]), (box[1][1], box[2][1]), color=color[i])
        ax1.plot((box[2][0], box[3][0]), (box[2][1], box[3][1]), color=color[i])
        ax1.plot((box[3][0], box[0][0]), (box[3][1], box[0][1]), color=color[i])

    ax1.imshow(img, cmap=plt.cm.gray)
    plt.show()


def window_extract_hands(config):
    if not os.path.isdir(config['datapath']+"/positive"):
        os.mkdir(config['datapath']+"/positive")
    if not os.path.isdir(config['datapath']+"/negative"):
        os.mkdir(config['datapath']+"/negative")
    if not os.path.isdir(config['datapath']+"/annotations_labels"):
        os.mkdir(config['datapath']+"/annotations_labels")
    h,w = config['slidingWindowHeight'],config['slidingWindowWidth']
    randomThresholdNegativeClass = 0.0012


    for file in os.listdir(config['datapath']+'/images'):

        img = skimage.io.imread(config['datapath']+'/images/'+file)
        annotations = load_own_json_annotation(config['datapath']+"/annotations_json/"+file[:-4]+".json")
        angles = get_bounding_box_angles(annotations)

        for i,(box,angle) in enumerate(zip(annotations,angles)):

            box = np.array(box)
            clipped_img = skimage.transform.rotate(img,angle+90)
            box_width = np.linalg.norm((box[1] - box[0]))
            box_height = np.linalg.norm((box[2] - box[1]))
            scale_factor = (w / box_width + h / box_height) / 2
            clipped_img = skimage.transform.rescale(clipped_img, scale_factor)
            rotate_scale_bounding_box(box, 270 - angle, np.array((img.shape[1] / 2 - 0.5, img.shape[0] / 2 - 0.5)), scale_factor)

            for y in range(0, clipped_img.shape[0]-h, config['slidingStep']):
                for x in range(0, clipped_img.shape[1]-w, config['slidingStep']):

                    xe = x + w
                    ye = y + h
                    annot_box = (box[2],box[0])
                    window_box = ((x,y),(xe,ye))

                    if overlap(annot_box,window_box) > 0.8:

                        #plot_bounding_box(clipped_img,[box,((xe,ye),(x,ye),(x,y),(xe,y))],color=['r','b'])
                        save_img = skimage.transform.resize(clipped_img[y:ye,x:xe], (config['slidingWindowHeight'],config['slidingWindowWidth']))
                        skimage.io.imsave(config['datapath']+"/positive/"+file[:-4]+"["+str(i)+str(x)+str(y)+"].png", save_img)
                        with open(config['datapath']+"/annotations_labels/"+file[:-4]+"["+str(i)+str(x)+str(y)+"].json",'w') as fout:
                            json.dump({'label':1},fout)

                    elif overlap(annot_box,window_box) < 0.2 and random.random()<randomThresholdNegativeClass and np.count_nonzero(clipped_img[y:ye,x:xe]==0)/(h*w)<0.3:

                        save_img = skimage.transform.resize(clipped_img[y:ye, x:xe], (config['slidingWindowHeight'], config['slidingWindowWidth']))
                        skimage.io.imsave(config['datapath']+"/negative/"+file[:-4]+"["+str(i)+str(x)+str(y)+"negative].png", save_img)
                        with open(config['datapath']+"/annotations_labels/"+file[:-4]+"["+str(i)+str(x)+str(y)+"negative].json",'w') as fout:
                            json.dump({'label':0}, fout)


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


if __name__ == '__main__':
    with open("config.json") as fin:
        config = json.load(fin)
    window_extract_hands(config)