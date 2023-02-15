import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# global parameters
gamma = 0.8
train_data_num = 1102
test_data_num = 1107           
bgrLower = np.array([0, 125, 125])      # 抽出するBGR色空間の下限
bgrUpper = np.array([255, 255, 255])    # 抽出するBGR色空間の上限


# Function to create some ordered images
def create_ordered_img(ordered_processes, datasets_path):
    '''
    ordered_processesで渡された文字列リストを参照し、対応するメソッドを実行

    Asrguments
    ----------
    ordered_processed : list
        行いたい画像処理の内容を格納し, リストの順番は画像処理の順番に対応.
        ex) ['gamma', 'edge', 'yellow', 'f-light']
    datasets_path : str
        読み込むデータセットのアドレス. 以下のex)を参考にする.
        ex) './dataset/train_images/train_'
    '''
    # create directories if they don't exist.
    os.makedirs('./preprocessed_2', exist_ok=True)
    # os.makedirs('./preprocessed/edge', exist_ok=True)
    # os.makedirs('./preprocessed/gamma', exist_ok=True)
    # os.makedirs('./preprocessed/yellow', exist_ok=True)
    # os.makedirs('./preprocessed/flat_lightness', exist_ok=True)

    # read dataset and call the ordered process.
    for i in range(test_data_num):
        if i < 10:
            path_num = '000' + str(i)
        elif i < 100:
            path_num = '00' + str(i)
        elif i < 1000:
            path_num = '0' + str(i)
        else:
            path_num = str(i)

        # read img file in dataset.
        img = cv2.imread(datasets_path + path_num + '.jpg')
        if img is None:
            print('"{}" is None.'.format(datasets_path + path_num + '.jpg'))
            exit()

        # call the ordered process.
        for index, process in enumerate(ordered_processes):
            if 'gamma' in process:
                img = gamma_correction(img)
            elif 'edge' in process:
                img = detect_edge(img)
            elif 'yellow' in process:
                img = extraction_color_of_lemon(img)
            elif 'f-light' in process:
                img = flatten_lightness(img)
            else:
                print("無効なprocessがordered_processesに含まれていました.\nvalue[{}] : {}".format(index, process))
                exit()

        # save img in 'preprocessed' directory.
        fig = plt.figure(frameon = False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig('./preprocessed/test_2_' + path_num + '.jpg')
        plt.close()


# Gamma correction
def gamma_correction(img, gam):
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # max pixel value
    imax = gray.max()
    # Create look-up table
    lookup_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
	    lookup_table[i][0] = imax * pow(float(i) / imax, 1.0 / gam)
    # Calculate with look-up table 
    return cv2.LUT(gray, lookup_table)


# Detect lemon and remove background
def detect_edge(
    img,
    BLUR = 21,
    CANNY_THRESH_1 = 15,            # TODO このパラメータの意味がわからない
    CANNY_THRESH_2 = 50,           # TODO このパラメータの意味がわからない
    MASK_DILATE_ITER = 10,
    MASK_ERODE_ITER = 10,
    MASK_COLOR = (0.0, 0.0, 1.0),
):
    # 0963, 0026, 0095, 0290, 0453, 0651, 0662, 0673,
    # 0006, 0007, 0008, 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]                                           # TODO contour[]のindexを適切にする？

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    # plt.imshow(mask_stack, cmap='gray')
    # plt.show()

    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
    c_blue, c_green, c_red = cv2.split(img)
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
    # plt.imshow(img_a)
    # plt.show()
    
    return img_a


# Extract colors near yellow with RGB
def extraction_color_of_lemon(img):
    img_mask = cv2.inRange(img, bgrLower, bgrUpper)           # create mask from RGB
    bgrResult = cv2.bitwise_and(img, img, mask = img_mask)    # mergge mask and source image
    
    return bgrResult


# Flattening the V histogram in HSV color space
def flatten_lightness(img):
    # convert from BGR to HSV 
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # flatten histgram of V value(= lightness) 
    hsv_img[:,:,2] = cv2.equalizeHist(hsv_img[:,:,2])
    # print histgram of V value after faltten
    # hist = cv2.calcHist([hsv_img], [2], None, [256], [0,256])
    # plt.plot(hist, color = "g")
    # plt.show()
    # convert from HSV to BGR
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    return bgr_img


# TODO 前処理後の画像をtest dataの600*600のサイズに合わせる


if __name__ == '__main__': 
    order = ['yellow', 'edge']
    path = './dataset/test_images_2/test_2_'
    create_ordered_img(order, path)
