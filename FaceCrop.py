import cv2
import os
import numpy as np

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    #print(img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)

    return img

def facecrop(image,imgname, interpolation = False, downsample_factor = None,
             guassianblurring = False, blurring_factor = None,
             pixelation=False, pixelation_factor = None,
             resize=False, resize_factor = None):

    # style 1: pure interpolation, downsampling factors can be 2,4,8
    # style 2: pure Guassian Blurring. sigma can be 1,2,4
    # style 3: Guassian Blurring and interpolation.

    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = load_image(image)
    #print(img.shape)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    if(len(faces) == 0): return None
    for f in faces:

        x, y, w, h = [v for v in f ]

        # x = int(x + 0.1*w)
        # w = int(0.75*w)
        #print('face height', h)
        #print('face width', w)
        img_1 = img[y:y+h,x:x+w]
        width = 128
        height = 128
        image_128 = cv2.resize(img_1,(128,128),interpolation=cv2.INTER_LANCZOS4)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1000, 1000)
        # cv2.imshow('image', image_128)
        # cv2.waitKey(0)

        if guassianblurring:
            img_1 = cv2.GaussianBlur(img_1,(5,5),sigmaX=blurring_factor,sigmaY=blurring_factor)

        if interpolation:
            img_downsample = cv2.resize(img_1,(int(w/downsample_factor), int(h/downsample_factor)),
                                        interpolation=cv2.INTER_LINEAR)
            img_1 = cv2.resize(img_downsample,(w,h), interpolation=cv2.INTER_CUBIC)
        if pixelation:
            img_downsample = cv2.resize(img_1,(int(width/pixelation_factor), int(height/pixelation_factor)),
                                        interpolation=cv2.INTER_LINEAR)
            img_1 = cv2.resize(img_downsample,(w,h), interpolation=cv2.INTER_NEAREST)
        if resize:
            img_1 = cv2.resize(img_1, (int(width / resize_factor), int(height / resize_factor)),
                                        interpolation=cv2.INTER_LINEAR)
        # img[y:y + h, x:x + w] = img_1
        #
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        #
        # sub_face = img[y:y+h, x:x+w]
        # face_file_name = "faces/" + imgname + "_" + str(y) + ".jpg"
        # cv2.imwrite(face_file_name, sub_face)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_1, image_128

def datasetGenerator(dir, folder, interpolation = False, downsample_factor = None,
             guassianblurring = False, blurring_factor = None,
             pixelation=False, pixelation_factor = None, sample_number=200000):
    '''
    We will use haarcascade classifier to detect and crop face image from celebA dateset.

    :param dir:   String: source image folder location.
    :param folder:    String: dataset name, will be as target folder name
    :param interpolation:   Bool: determine whether bicubic interpolation or not
    :param downsample_factor:  int: (2,4,8) downsampling factor
    :param guassianblurring:   Bool: determine whether Guassian Blurring or not
    :param blurring_factor:   int: (1,2,4,8) Variance of Guassian Filter
    :param pixelation:     Bool: determine whether pixelation or not
    :param pixelation_factor:   int: (2,4,8) downsampling factor
    :param sample_number:   Int: number of samples
    :return:  None
    '''

    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    #os.mkdir(folder)

    for ii in range(sample_number):

        index = "%06d" % (ii+1)
        image_directory = dir + index + '.jpg'
        print(image_directory)

        img = cv2.imread(image_directory)
        minisize = (img.shape[1], img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)
        if(len(faces) == 0): continue
        x, y, w, h = [v for v in faces[0]]  # only need the first detected face image

        img_raw = img[y:y + h, x:x + w]
        img_raw = cv2.resize(img_raw, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        img_1 = img_raw
        #print(img_1.shape)
        if guassianblurring:
            img_1 = cv2.GaussianBlur(img_1,(5,5),sigmaX=blurring_factor,sigmaY=blurring_factor)
        if interpolation:
            img_downsample = cv2.resize(img_1,(int(128/downsample_factor), int(128/downsample_factor)),
                                        interpolation=cv2.INTER_LINEAR)
            img_1 = cv2.resize(img_downsample,(128,128), interpolation=cv2.INTER_CUBIC)
        if pixelation:
            img_downsample = cv2.resize(img_1,(int(128/downsample_factor), int(128/downsample_factor)),
                                        interpolation=cv2.INTER_NEAREST)
            img_1 = cv2.resize(img_downsample,(128,128), interpolation=cv2.INTER_NEAREST)

        #print(img_1.shape)
        write_dir = os.path.join(folder, index + '.png')
        cv2.imwrite(write_dir, img_1)
        origin_dir = os.path.join(folder, index + '.png')
        cv2.imwrite(origin_dir, img_raw)


dir = './data/celeba/img_align_celeba/'
small_dir = './data/celeba/small/'
origin_dir = './data/celeba/origin/'
num = 1
for ii in range(2,200000):
    if(ii %100 == 0): print(ii)
    name = "%06d" % ii
    suffix = '.jpg'
    image_path = dir+name+suffix
    print(image_path)
    result = facecrop(image_path,'1', resize=True, resize_factor=4)
    if result == None:
        continue
    small_image, origin_image = result
    filename = "%06d" % num
    store_path = small_dir + filename + suffix
    #print(store_path)
    cv2.imwrite(store_path, small_image)
    filename = "%06d" % num
    store_path = origin_dir + filename + suffix
    # print(store_path)
    cv2.imwrite(store_path, origin_image)
    num = num + 1
