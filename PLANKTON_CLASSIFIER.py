# import section
import matplotlib
matplotlib.use('Agg')
from skimage import feature
from itertools import cycle
from sklearn.metrics import confusion_matrix
from keras import Model
from scipy import interp
import random
from mahotas.zernike import zernike_moments
from mahotas.features import haralick
from keras.optimizers import Adam
import cv2
import keras
import copy
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from keras.layers import Lambda
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1
import os
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, concatenate
import numpy as np
from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans


class PLANKTON_CLASSIFIER():

    def normalize_test_train_for_newclasses(self, features, x_test, name):

        medie_per_norm = np.loadtxt(self.address + 'TRAINED DETECTORS/medie' + name + '.csv')
        medie_per_std = np.loadtxt(self.address + 'TRAINED DETECTORS/medie1' + name + '.csv')

        for i in range(0, features):
            x_test[:, i] = (x_test[:, i] - medie_per_std[i]) / (-medie_per_std[i] + medie_per_norm[i])

        return x_test

    class LocalBinaryPatterns:
        def __init__(self, numPoints, radius):
            # store the number of points and radius
            self.numPoints = numPoints
            self.radius = radius



        def describe(self, image, eps=1e-7):
            # compute the Local Binary Pattern representation
            # of the image, and then use the LBP representation
            # to build the histogram of patterns
            lbp = feature.local_binary_pattern(image, self.numPoints,
                                               self.radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, self.numPoints + 3),
                                     range=(0, self.numPoints + 2))

            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            # return the histogram of Local Binary Patterns
            return hist

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((l / 255.0) ** invGamma) * 255
                          for l in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    # put here the address.
    def image_processor(self, address):

        train_folder = address + 'TRAIN_IMAGE/'
        test_folder = address + 'TEST_IMAGE/'
        output_segmentation_train = address + 'BIN_TRAIN_IMAGE/'
        output_segmentation_test = address + 'BIN_TEST_IMAGE/'

        if not os.path.isdir(output_segmentation_test):
            os.mkdir(output_segmentation_test)

        if not os.path.isdir(output_segmentation_train):
            os.mkdir(output_segmentation_train)

        files = self.files

        mydir2 = test_folder

        for j in range(0, len(files)):

            if files[j] != ".DS_Store":
                if not os.path.isdir(output_segmentation_train + files[j]):
                    os.mkdir(output_segmentation_train + files[j])

                files2 = os.listdir(train_folder + '/' + files[j])

                a44 = 100
                if len(files2) <= a44:
                    a44 = len(files2)
                for aa in range(0, a44):

                    if files2[aa] != ".DS_Store":

                        img = cv2.imread(train_folder + '/' + files[j] + '/' + files2[aa])
                        img = cv2.medianBlur(img, 7, img)
                        #
                        try:

                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                         15, 2)

                            img2 = cv2.bitwise_not(img2)
                            img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE,
                                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

                        except:
                            pass

                        try:
                            img = cv2.imread(train_folder + '/' + files[j] + '/' + files2[aa])
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                            img = cv2.bitwise_not(img)
                            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

                            img = cv2.convertScaleAbs(img)
                            im3 = copy.copy(img)
                            img = cv2.bitwise_or(img, img2)
                            # edge detection

                            im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            z = 0
                            frame4 = img
                            imy = np.zeros_like(im3)
                            ar_max = 0
                            ar_max2 = 0
                            for i in x:
                                # minimum rectangle containing the object
                                PO2 = cv2.boundingRect(i)
                                area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                # LET'S CHOOSE THE AREA THAT WE WANT
                                if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

                                    moments = cv2.moments(i)
                                    cv2.drawContours(imy, [i], -1, (255, 0, 0), -1)

                                    ar = moments['m00']
                                    if ar > ar_max2:
                                        ar_max = i
                                        ar_max2 = ar

                            try:

                                i = ar_max
                                PO2 = cv2.boundingRect(i)
                                imx = np.zeros_like(im3)
                                cv2.drawContours(imx, [i], -1, (255, 0, 0), -1)
                                o = imx
                                ttemp = copy.copy(im3)
                                ttemp[ttemp != 0] = 1
                                o[o != 0] = 1
                                imy[imy != 0] = 1
                                if np.sum(o) < np.sum(imy) * 0.5:
                                    ############################################
                                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                                    im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                    z = 0
                                    frame4 = img

                                    ar_max = 0
                                    ar_max2 = 0
                                    for i in x:
                                        # minimum rectangle containing the object
                                        PO2 = cv2.boundingRect(i)
                                        area = cv2.countNonZero(
                                            frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                        # LET'S CHOOSE THE AREA THAT WE WANT
                                        if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

                                            moments = cv2.moments(i)
                                            ar = moments['m00']
                                            if ar > ar_max2:
                                                ar_max = i
                                                ar_max2 = ar

                                    try:
                                        i = ar_max
                                        PO2 = cv2.boundingRect(i)
                                        optim_contour = np.zeros_like(im3)
                                        area = cv2.countNonZero(
                                            frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                        cv2.drawContours(optim_contour, [i], -1, (255, 255, 0), -1)
                                        #######################display objects###########################ll
                                        if np.sum(optim_contour) > 255 * np.shape(optim_contour)[1] * \
                                                np.shape(optim_contour)[0] - 500:
                                            continue
                                        cv2.imwrite(output_segmentation_train + '/' + files[j] + '/' + files2[aa],
                                                    optim_contour)
                                        continue
                                    except:
                                        pass

                                ############################################

                                area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                optim_contour = np.zeros_like(im3)
                                area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                cv2.drawContours(optim_contour, [i], -1, (255, 0, 255), -1)

                                #######################display objects###########################ll

                                if np.sum(optim_contour) > 255 * np.shape(optim_contour)[1] * np.shape(optim_contour)[
                                    0] - 500:
                                    continue

                                cv2.imwrite(output_segmentation_train + '/' + files[j] + '/' + files2[aa],
                                            optim_contour)
                            except:
                                pass
                        except:
                            pass

    def feature_extractor(self, address, train_folder, train_folder_bin, files):

        train_folder = address + 'TRAIN_IMAGE/'
        train_folder_bin = address + 'BIN_TRAIN_IMAGE/'
        train_features_bin_OUTPUT = address + 'TRAIN_features_res/'

        if not os.path.isdir(train_features_bin_OUTPUT):
            os.mkdir(train_features_bin_OUTPUT)

        total = []

        count = np.zeros((10))

        for j in range(0, len(files)):
            if files[j] != ".DS_Store":
                conteggio = 0

                if not os.path.isdir(train_features_bin_OUTPUT + files[j]):
                    os.mkdir(train_features_bin_OUTPUT + files[j])

                files2 = os.listdir(train_folder_bin + files[j])
                # files2 = total[np.where( files44 == files[j])[0][0]]

                for aa in range(0, len(files2)):

                    if files2[aa] != ".DS_Store":

                        img = cv2.imread(train_folder_bin + '\\' + files[j] + '\\' + files2[aa])

                        try:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        except:
                            continue
                            pass

                        im3 = img

                        # ZERNIKE MOMENTS

                        list = np.zeros((133, 1), dtype=float)

                        W, H = im3.shape
                        R = min(W, H) / 2
                        a = zernike_moments(im3, R, 8)
                        try:

                            im5 = cv2.imread(train_folder + '/' + files[j] + '/' + files2[aa])
                            im4 = copy.copy(im5)

                            if np.shape(im5)[0] != np.shape(im3)[0]:
                                deltax = np.shape(im5)[0] - np.shape(im3)[0]

                                if deltax < 0:
                                    im5 = np.pad(im5, pad_width=((-deltax, 0), (0, 0)), mode='constant')

                                if deltax > 0:
                                    im3 = np.pad(im3, pad_width=((deltax, 0), (0, 0)), mode='constant')

                            if np.shape(im5)[1] != np.shape(im3)[1]:

                                deltay = np.shape(im5)[1] - np.shape(im3)[1]

                                if deltay > 0:
                                    im3 = np.pad(im3, pad_width=((0, 0), (0, deltay)), mode='constant')

                                if deltay < 0:
                                    im5 = np.pad(im5, pad_width=((0, 0), (0, -deltay)), mode='constant')

                            mask = np.zeros(im3.shape[:2], np.uint8)

                            mask[im3 == 0] = 255

                        except:
                            continue

                        try:

                            momentsintensity = np.zeros((8, 1))
                            imgray = copy.copy(im5)
                            imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
                            histogram = cv2.calcHist([imgray], [0], mask, [255], [0, 255])

                            momentsintensity[0] = histogram.mean()
                            momentsintensity[1] = histogram.std()
                            from scipy.stats import kurtosis
                            from scipy.stats import skew

                            momentsintensity[2] = skew(histogram)
                            momentsintensity[3] = kurtosis(histogram)
                            from scipy.stats import entropy

                            histogram = histogram / np.max(histogram)
                            momentsintensity[4] = entropy(histogram)

                            uuu = copy.copy(im5)

                            # uuu[:, :, 0] = cv2.bitwise_and(uuu[:, :, 0], mask)
                            # uuu[:, :, 1] = cv2.bitwise_and(uuu[:, :, 1], mask)
                            # uuu[:, :, 2] = cv2.bitwise_and(uuu[:, :, 2], mask)
                            #
                            # momentsintensity[5] = np.mean(uuu[:, :, 0]) / np.mean(uuu[:, :, 2])
                            # momentsintensity[6] = np.mean(uuu[:, :, 0]) / np.mean(uuu[:, :, 1])
                            # momentsintensity[7] = np.mean(uuu[:, :, 1]) / np.mean(uuu[:, :, 2])
                            im5[im3 == 0] = 255
                            im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
                            im4 = copy.copy(im5)
                            # im4 = cv2.medianBlur(im5, 3, dst=im4)
                            # SHAPE INDEX



                        except:
                            pass

                        try:
                            import matplotlib.pyplot as plt
                            from mpl_toolkits.mplot3d import Axes3D
                            from scipy import ndimage as ndi
                            from skimage.feature import shape_index
                            from skimage.draw import circle

                            # LOCAL BINARY PATTERNS

                            lbp = self.LocalBinaryPatterns(54, 8)
                            lbp = lbp.describe(im5)
                            # HARALICK FEATURES
                            b = haralick(im5)
                            b = b.mean(axis=0)
                            cont = 0
                            # edge_detection
                            im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            # ready for the list to save
                            z = 0
                            frame4 = img
                            ar_max = 0
                            ar_max2 = 0

                            for i in x:
                                # minimum rectangle containing the object
                                PO2 = cv2.boundingRect(i)
                                area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                                # LET'S CHOOSE THE AREA THAT WE WANT
                                if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

                                    moments = cv2.moments(i)

                                    ar = moments['m00']
                                    if ar > ar_max2:
                                        ar_max = i
                                        ar_max2 = ar

                        except:
                            pass

                        try:

                            i = ar_max
                            # minimum rectangle containing the object
                            PO2 = cv2.boundingRect(i)
                            area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                            # area must be computed in this way, many times contour can be opened.
                            PO = PO2

                            # LET'S CHOOSE THE AREA THAT WE WANT
                            if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

                                # extract some features
                                moments = cv2.moments(i)
                                if moments['m00'] != 0.0:
                                    #     # centroid
                                    xc = moments['m10'] / moments['m00']

                                    yc = moments['m01'] / moments['m00']
                                # cont += 1
                                humoments = cv2.HuMoments(moments)

                                list[cont, 0] = moments['m00']
                                cont += 1
                                list[cont, 0] = humoments[0]
                                cont += 1
                                list[cont, 0] = humoments[1]
                                cont += 1
                                list[cont, 0] = humoments[2]
                                cont += 1
                                list[cont, 0] = humoments[3]
                                cont += 1
                                list[cont, 0] = humoments[4]
                                cont += 1
                                list[cont, 0] = humoments[5]
                                cont += 1
                                list[cont, 0] = humoments[6]
                                cont += 1
                                list[cont, 0] = area
                                cont += 1
                                # perimeter
                                list[cont, 0] = cv2.arcLength(i, True)
                                cont += 1
                                # equivalent diameter
                                list[cont, 0] = np.sqrt(4 * list[0] / np.pi)
                                cont += 1
                                ellip = cv2.fitEllipse(i)
                                (center, axes, orientation) = ellip
                                minor_axis = min(axes)
                                major_axis = max(axes)
                                list[cont, 0] = np.sqrt(1 - (minor_axis * minor_axis) / (major_axis * major_axis))
                                # length of MAJOR and minor axis of fitting ellipse
                                cont += 1
                                list[cont, 0] = minor_axis
                                cont += 1
                                list[cont, 0] = major_axis
                                cont += 1
                                # roundness
                                list[cont, 0] = (4 * np.pi * list[0]) / (
                                        cv2.arcLength(i, True) * cv2.arcLength(i, True))
                                cont += 1
                                # color information
                                width = PO[3]
                                height = PO[2]
                                # shape factor as percentace of occupancy
                                list[cont, 0] = area / (width * height)
                                cont += 1
                                list[cont, 0] = (moments['m00'] * moments['m00']) / (
                                        2 * np.pi * (moments['mu20'] + moments['mu02']))
                                cont += 1
                                rect = cv2.minAreaRect(i)
                                box = cv2.boxPoints(rect)
                                x1 = box[0, 0]
                                y1 = box[0, 1]
                                x2 = box[1, 0]
                                y2 = box[1, 1]
                                x3 = box[2, 0]
                                y3 = box[2, 1]
                                tocontinue = 0
                                # rectangluar
                                try:
                                    list[cont, 0] = moments['m00'] / (
                                            np.abs((rect[1][0] - rect[0][0])) * np.abs((rect[1][1] - rect[0][1])))
                                    if np.abs((rect[1][0] - rect[0][0])) * np.abs((rect[1][1] - rect[0][1])) == 0:
                                        tocontinue = 1
                                    cont += 1

                                    convex = cv2.convexHull(i)
                                    list[cont, 0] = cv2.arcLength(i, True) / cv2.arcLength(convex, True)
                                    cont += 1
                                    if cv2.arcLength(convex, True) == 0:
                                        tocontinue = 1

                                except:
                                    pass

                                if tocontinue == 1:
                                    continue
                                # solidity
                                widthmin = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                                heightmin = np.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))
                                if widthmin > heightmin:
                                    list[cont, 0] = widthmin / heightmin
                                    cont += 1
                                    # higher dimension
                                    list[cont, 0] = widthmin
                                    cont += 1
                                    # lower dinension
                                    list[cont, 0] = heightmin
                                    cont += 1
                                else:
                                    list[cont, 0] = heightmin / widthmin
                                    cont += 1
                                    # higher dimension
                                    list[cont, 0] = heightmin
                                    cont += 1
                                    # lower dinension
                                    list[cont, 0] = widthmin
                                    cont += 1
                                for newindex in range(2, len(a)):
                                    list[cont, 0] = a[newindex]
                                    cont += 1
                                for newindex in range(0, len(lbp)):
                                    list[cont, 0] = lbp[newindex]
                                    cont += 1
                                for newindex in range(0, len(b)):
                                    list[cont, 0] = b[newindex]
                                    cont += 1
                                for newindex in range(0, len(momentsintensity)):
                                    list[cont, 0] = momentsintensity[newindex]
                                    cont += 1
                                points = i
                                s = np.zeros((len(points), 1), dtype=np.complex64)
                                count = 0

                                xc2 = 1 / len(points) * np.sum(points[:, 0, 0])

                                yc2 = 1 / len(points) * np.sum(points[:, 0, 1])
                                for p in points:
                                    # s[count, 0] = ((p[0][0] + (p[0][1]*1j)))
                                    s[count, 0] = ((p[0][0] - xc2) ** 2 + (p[0][1] - yc2) ** 2) ** 0.5
                                    count += 1
                                fourier = np.fft.fft(s)
                                rho = np.abs(fourier)
                                rho = rho / rho[0]
                                rho = rho[0:11]
                                medd = np.mean(s)
                                stddd = np.std(s)
                                AAA = np.fft.ifft(fourier)
                                plt.plot(AAA)
                                for newindex in range(0, len(rho)):
                                    list[cont, 0] = rho[newindex]
                                    cont += 1
                                if conteggio < 1500:
                                    np.savetxt(
                                        train_features_bin_OUTPUT + '\\' + files[j] + '\\' + files2[conteggio] + '.csv',
                                        list)
                                    conteggio += 1

                        except:
                            pass

    def euclidian_distance(self, a, b):
        c = b
        for i in range(0, np.shape(a)[0] - 1):
            c = np.row_stack((c, b))
        dist = np.linalg.norm(a - c, axis=1)
        return dist

    # function needed to normalize the test set in the same way than the training, using training max and min value
    def normalize_test_train(self, features, X, x_test):

        medie_per_norm = np.zeros((features, 1), dtype=np.float)
        medie_per_std = np.zeros((features, 1), dtype=np.float)

        for i in range(0, features):
            medie_per_norm[i] = float(np.max(X[:, i]))
            medie_per_std[i] = float(np.min(X[:, i]))

            # train normalization
            X[:, i] = (X[:, i] - medie_per_std[i]) / (-medie_per_std[i] + medie_per_norm[i])

        for i in range(0, features):
            # test normalization
            x_test[:, i] = (x_test[:, i] - medie_per_std[i]) / (-medie_per_std[i] + medie_per_norm[i])

        # let us save these values for further analysis

        return X, x_test

    # we use this normalization for preprocessing and PCA analysis
    def normalize(self, X):

        for i in range(0, np.shape(X)[1]):
            if np.sum(X[:, i]) != 0 and np.sum(X[:, i]) != (np.shape(X[:, i])[0] * X[0, i]):
                X[:, i] = (X[:, i] - np.min(X[np.nonzero(X[:, i]), i])) / (
                        np.max(X[np.nonzero(X[:, i]), i]) - np.min(X[np.nonzero(X[:, i]), i]))
        return X

    def reading(self, address):
        train_folder = address + 'TRAIN_FEATURES/'
        mydir = train_folder
        test_folder = address + 'TEST_FEATURES/'
        files = os.listdir(train_folder)
        labels = list()
        mydir2 = test_folder
        global files3
        try:
            files3 = os.listdir(mydir2)


        except:
            files3=[]
        cont2 = 0

        # number of species
        cont3 = 0
        x_train = np.zeros((133, self.alfa * self.dim), dtype=np.float)
        y_train = np.zeros((self.alfa * self.dim, self.alfa), dtype=np.int)

        x_test = np.zeros((133, self.alfa * self.dim2), dtype=np.float)
        y_test = np.zeros((self.alfa * self.dim2, self.alfa), dtype=np.int)

        target = np.zeros((self.alfa * self.dim), dtype=np.int)
        species = np.zeros((self.alfa * self.dim), dtype=np.str)

        cont = 0
        # reading training data
        for j in range(0, len(files)):

            if files[j] != ".DS_Store":

                files_TRAIN = os.listdir(mydir + '/' + files[j])

                labels.append(files[j])
                aa = 0
                # flag for reading

                for i in range(0, len(files_TRAIN)):

                    if files_TRAIN[i] != ".DS_Store" and aa < self.dim:

                        file_read = open(mydir + files[j] + '/' + files_TRAIN[aa], 'r')

                        try:

                            x_train[:, cont2] = np.loadtxt(file_read)

                            y_train[cont2, cont] = 1
                            cont2 += 1
                            aa += 1
                            target[cont2] = j
                            species[cont2] = files[j]
                        except:
                            pass

                cont += 1
        # reading test data
        cont = 0

        if self.static==0:

            for j in range(0, len(files)):

                if files[j] != ".DS_Store":

                    files_TEST = os.listdir(mydir2 + '/' + files[j])
                    # flag for reading
                    aa = 0

                    for i in range(0, len(files_TEST)):

                        if files_TEST[i] != ".DS_Store" and aa < self.dim2:

                            file_read = open(mydir2 + files[j] + '/' + files_TEST[aa], 'r')

                            try:

                                x_test[:, cont3] = np.loadtxt(file_read)
                                y_test[cont3, cont] = 1
                                cont3 += 1
                                aa += 1
                            except:
                                pass
                    cont += 1

            x_train = np.transpose(x_train)

            x_test = np.transpose(x_test)

            x_train = np.delete(x_train, 122, axis=1)

            x_test = np.delete(x_test, 122, axis=1)

            x_train = x_train[:, :-1]
            x_test = x_test[:, :-1]

        else:
            x_train = np.transpose(x_train)
            x_train = np.delete(x_train, 122, axis=1)

            x_train = x_train[:, :-1]


        import copy

        y_train = y_train[0:self.alfa * self.dim, 0:self.alfa]
        y_test = y_test[0:self.alfa * self.dim2, 0:self.alfa]

        np.savetxt(self.address + 'x_train_for_further_code.csv', x_train,
                   delimiter=",")
        np.savetxt(self.address + 'x_test_for_further_code.csv', x_test,
                   delimiter=",")

        return x_train, y_train, x_test, y_test, labels

    def evaluate_purity(self, k, predicted, Y, size):

        score_best = np.zeros((k, k))

        for i in range(0, k):
            for j in range(0, k):
                # computing all the possible overlap
                score_best[i, j] = np.count_nonzero(Y[predicted[i], j] == 1)

        # taking the maximum arguments and values for accuracy
        master = np.argmax(score_best, axis=0)
        records_array = master
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                           return_index=True)

        # sets of indices

        res = np.split(idx_sort, idx_start[1:])

        # filter them with respect to their size, keeping only items occurring more than once

        vals = vals[count > 1]
        a = vals
        a = np.unique(a)

        specie_over = list()

        for j in range(0, len(a)):
            specie_over.append(np.where(master == a[j]))
        maxs = np.max(score_best, axis=1)

        # maxs = np.max(score_best, axis=0)
        master2 = np.argmax(score_best, axis=1)

        purit = np.sum(maxs) / size
        return master2, purit, score_best, specie_over

    def evaluate_purity_OVERLAP(self, k, predicted, Y, size):

        score_best = np.zeros((k, k))
        for i in range(0, k):
            for j in range(0, k):
                # computing all the possible overlap
                score_best[i, j] = np.count_nonzero(Y[predicted[i], j] == 1)
        # taking the maximum arguments and values for accuracy
        master = np.argmax(score_best, axis=0)
        records_array = master
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                           return_index=True)
        # sets of indices
        res = np.split(idx_sort, idx_start[1:])
        # filter them with respect to their size, keeping only items occurring more than once
        specie_over = list()
        vals = vals[count > 1]
        a = vals
        a = np.unique(a)
        master2 = np.argmax(score_best, axis=1)

        for i in range(0, len(a)):
            specie_over.append(np.where(master == a[i])[0])
        # this is referred to species
        maxs = np.max(score_best, axis=0)
        # this is referred to clusters
        maxs2 = np.max(score_best, axis=1)
        cluster_totals = np.asarray(list(range(0, self.alfa)))
        macrocluster = list()
        species_overlapped = list()

        self.num_species = self.alfa - len(specie_over)

        for i in range(0, len(cluster_totals)):
            if not i in master2:
                maxs2[master[i]] += maxs2[master[i]]
                macrocluster.append(master[i])
                species_overlapped.append(i)
            # maxs = np.max(score_best, axis=0)
        purit = np.sum(maxs2) / size
        return master2, purit, score_best, specie_over, macrocluster, species_overlapped

    def isdifferente(self, x111):
        elements = np.zeros((len(x111), 1)) - 1
        num = 0
        for i in range(0, len(x111)):
            if elements[x111[i]] == -1:
                elements[x111[i]] = i
            else:
                num += 1
        return num

    def PCA_custom(self, x_def, y_def):

        #####################

        if self.static ==1:
            x_def = np.column_stack((x_def[:,0:118],x_def[:,122:]))
        ##################################### DELETE!
        from sklearn import preprocessing
        from sklearn.decomposition import PCA
        # x_def, y_def = self.binning(5,x_def,y_def)
        std_scale = preprocessing.StandardScaler().fit(x_def)
        x_def = std_scale.transform(x_def)

        # PCA from scikit-learn
        pca = PCA(n_components=np.shape(x_def)[1])
        x_def[np.isnan(x_def)] = 0
        x_def[np.isinf(x_def)] = 0

        pca.fit(x_def)
        X = pca.transform(x_def)
        files = self.files
        labels = files

        eigenvectors = pca.components_
        var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)
        plt.plot(var)
        plt.style.context('seaborn-whitegrid')
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(30, 100.5)
        num_species = self.alfa
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn import preprocessing

        # this is beacuse we want to use MYC+RAS as testing data, thus, I fix the trains, but only for the NN not for the PCA
        y = y_def
        z = np.zeros((len(y), 1))

        for i in range(0, num_species):
            z[y[:, i] == 1, :]

        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, self.alfa))
        c = colors
        plt.figure()

        for i in range(0, num_species):
            plt.scatter(X[y[:, i] == 1, :][:, 0], X[y[:, i] == 1, :][:, 1], c=c[i], edgecolor='k', label=labels[i])

        plt.legend()
        plt.ylabel('PC1')
        plt.xlabel('PC0')
        plt.figure()
        for i in range(0, num_species):
            plt.scatter(X[y[:, i] == 1, :][:, 0], X[y[:, i] == 1, :][:, 2], c=c[i], edgecolor='k', label=labels[i])

        plt.legend()
        plt.ylabel('PC0')
        plt.xlabel('PC2')
        plt.figure()

        for i in range(0, num_species):
            plt.scatter(X[y[:, i] == 1, :][:, 1], X[y[:, i] == 1, :][:, 2], c=c[i], edgecolor='k', label=labels[i])
        plt.ylabel('PC1')
        plt.xlabel('PC2')

        plt.legend()
        fig = plt.figure()
        ax = Axes3D(fig)

        for i in range(0, num_species):
            ax.scatter(X[y[:, i] == 1, :][:, 0], X[y[:, i] == 1, :][:, 1], X[y[:, i] == 1, :][:, 2], c=c[i],
                       edgecolor='k', label=labels[i])
        ax.w_xaxis.set_ticklabels(['PC 1'])
        ax.w_yaxis.set_ticklabels(['PC 2'])
        ax.w_zaxis.set_ticklabels(['PC 3'])
        ax.legend(ncol=2)

        plt.show()
        if self.static==0:
            self.m = 1.6
        else:
            self.m = 1.15

        return X, eigenvectors

    def clusters_comp(self, Train_original):
        import statistics

        maxssss3 = list()

        for t in range(0, 10):
            X2 = copy.copy(Train_original)
            X2 = X2[~np.all(X2 == 0, axis=1)]
            X2[np.isnan(X2)] = 0
            X2[np.isinf(X2)] = 0
            X2 = self.normalize(X2)
            X2[np.isnan(X2)] = 0
            X2[np.isinf(X2)] = 0
            partition_entropy = list()

            for azz in range(2, 12):
                X3 = copy.copy(X2)
                k = azz
                fuzzy_kmeans = FuzzyKMeans(k=azz, m=1.4)
                fuzzy_kmeans.fit(X3)
                #PARTITION ENTROPY DEFINITION
                F2 = 1 / np.shape(fuzzy_kmeans.fuzzy_labels_)[0] * np.sum(
                    (np.sum(fuzzy_kmeans.fuzzy_labels_ * np.log(fuzzy_kmeans.fuzzy_labels_), axis=1)))
                partition_entropy.append((F2 - 1 / azz) / (1 - 1 / azz))
            maxssss3.append(np.argmax(np.asarray(partition_entropy)) + 2)

        try:
            print('num_clusters = ' + str(statistics.mode(partition_entropy)))
        except:
            if len(np.unique(partition_entropy))==0:
                #very robust estimation of number of clusters
                print('num_clusters = ' + str(partition_entropy[0]))

    def unsupervised_partitioning(self, x_train, y_train, X):
        import random

        Train_original = copy.copy(x_train)


        X2 = copy.copy(Train_original)

        X3 = copy.copy(X2)


        dim_Train = self.dim_Train
        dim_Test = self.dim - self.dim_Train
        X_train = np.zeros((dim_Train * self.alfa, 131))
        Y_train = np.zeros((dim_Train * self.alfa, self.alfa))
        for i in range(0, self.alfa):
            X_train[i * dim_Train:i * dim_Train + dim_Train, :] = X3[i * self.dim:i * self.dim + dim_Train, :]
            Y_train[i * dim_Train:i * dim_Train + dim_Train, :] = y_train[i * self.dim:i * self.dim + dim_Train, :]

        self.clusters_comp(X_train)

        self.X7 = copy.copy(X_train)
        X3 = copy.copy(X_train)
        self.y_train = y_train

        y_train = copy.copy(Y_train)


        fuzzy_kmeans = FuzzyKMeans(k=self.alfa, m=self.m)
        X2 = copy.copy(X3)
        # X2 = X2[:,0:120]
        if self.static==1:
            X2 = np.column_stack((X2[:,0:118],X2[:,122:]))

        X2[np.isnan(X2)] = 0

        X2[np.isinf(X2)] = 0

        X2 = X2[~np.all(X2 == 0, axis=1)]
        X2 = self.normalize(X2)
        X2[np.isnan(X2)] = 0

        X2[np.isinf(X2)] = 0

        # X2 = X2[~np.all(X2 == 0, axis=1)]

        try:
            fuzzy_kmeans.fit(X2)
        except:
            pass
        num_species = self.alfa
        k = self.alfa
        labels = list()
        for i in range(0, self.alfa):
            labels.append('cluster' + str(i))
        predicted = list()
        for p in range(0, k):
            predicted.append(list())
        for i in range(0, k):
            predicted[i].append(np.where(fuzzy_kmeans.labels_ == i))
        master, purit, scorororo, spe, macro, r3 = self.evaluate_purity_OVERLAP(num_species, predicted, y_train,
                                                                                np.shape(y_train)[0])

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, self.alfa))
        colorss = colors

        for i in range(0, k):
            ax.scatter(X[predicted[i], 0], X[predicted[i], 1], X[predicted[i], 2], c=colorss[master[i]], edgecolor='k',
                       label=labels[master[i]])
        plt.legend(ncol=2)
        ax.w_xaxis.set_ticklabels(['PC 1'])
        ax.w_yaxis.set_ticklabels(['PC 2'])
        ax.w_zaxis.set_ticklabels(['PC 3'])
        plt.figure()
        for i in range(0, num_species):
            plt.scatter(X[predicted[i], 0], X[predicted[i], 1], c=colorss[master[i]], edgecolor='k', label=labels[i])
        plt.legend()
        plt.figure()
        plt.ylabel('PC0')
        plt.xlabel('PC1')
        for i in range(0, num_species):
            plt.scatter(X[predicted[i], 1], X[predicted[i], 2], c=colorss[master[i]], edgecolor='k', label=labels[i])
        plt.legend()
        plt.ylabel('PC1')
        plt.xlabel('PC2')
        plt.figure()
        for i in range(0, num_species):
            plt.scatter(X[predicted[i], 0], X[predicted[i], 2], c=colorss[master[i]], edgecolor='k', label=labels[i])
        plt.legend()
        plt.ylabel('PC0')
        plt.xlabel('PC2')
        plt.legend()
        print(purit)
        print(self.isdifferente(master))
        plt.show()
        centroid = fuzzy_kmeans.cluster_centers_

        labels_true_for_Test = np.zeros((self.alfa * dim_Train, k))

        for aaa in range(0, self.alfa):
            labels_true_for_Test[predicted[aaa], master[aaa]] = 1

        y_train4 = copy.copy(self.y_train)

        for a in range(0, len(macro)):
            y_train4[np.where(y_train4[:, r3[a]] == 1)[0], macro[a]] = 1
            y_train4[np.where(y_train4[:, r3[a]] == 1)[0], r3[a]] = 0

        y_train4 = np.transpose(y_train4)
        labels_true_for_Test = np.transpose(labels_true_for_Test)

        labels_true_for_Test = labels_true_for_Test[~np.all(y_train4 == 0, axis=1)]

        y_train4 = y_train4[~np.all(y_train4 == 0, axis=1)]
        y_train4 = np.transpose(y_train4)
        labels_true_for_Test = np.transpose(labels_true_for_Test)

        self.y_train4 = y_train4

        return labels_true_for_Test

    def GMM(self, X2, X, y_train):
        from sklearn.mixture import GaussianMixture
        k = self.alfa
        X2 = normalize(X2)
        gmm = GaussianMixture(n_components=self.alfa)
        gmm.fit(X2)
        predicted_gmm = gmm.predict(X2)
        fig4 = plt.figure()
        ax = fig4.add_subplot(111, projection='3d')
        jet = plt.cm.jet
        labels = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7', 'cluster 8',
                  'cluster 9', 'cluster 10', 'cluster11', 'cluster12', 'cluster13', 'cluster14', 'cluster15', 'cluster 16',
                  'cluster 17', 'cluster 18', 'cluster19', 'cluster20', 'cluster21', 'cluster22']

        colorss = jet(np.linspace(0, 1, self.alfa))
        predicted1 = list()
        for p in range(0, k):
            predicted1.append(list())
        for i in range(0, k):
            predicted1[i].append(np.where(predicted_gmm == i))
        master, purit = self.evaluate_purity(k, predicted1, y_train, np.shape(X2)[0])

        for i in range(0, k):
            ax.scatter(X[predicted1[i], 0], X[predicted1[i], 1], X[predicted1[i], 2], c=colorss[master[i]], edgecolor='k',
                       label=labels[master[i]])
        plt.legend()
        ax.w_xaxis.set_ticklabels(['PC 1'])
        ax.w_yaxis.set_ticklabels(['PC 2'])
        ax.w_zaxis.set_ticklabels(['PC 3'])
        print(purit)
        print(self.isdifferente(master))
        return gmm

    def oneclassSVM(self, X2, y_train, x_test, y_test, address):
        accuracy = np.zeros((2, 10))

        from sklearn import svm

        for c in range(0, 10):

            x = X2[np.where(y_train[:, c] == 1), :][0]
            maxs = np.max(x, axis=0)
            mins = np.min(x, axis=0)
            x_t = copy.copy(x_test)
            if c != 9:
                x = np.row_stack((x))
            else:
                x = np.row_stack((x))

            for i in range(0, 131):
                x_t[:, i] = (x_t[:, i] - mins[i]) / (maxs[i] - mins[i])

                x[:, i] = (x[:, i] - mins[i]) / (maxs[i] - mins[i])

            clf = svm.OneClassSVM(kernel='rbf', degree=5, nu=0.1)
            clf.fit(x)

            a = clf.predict(x_t[c * 140:c * 140 + 140, :])
            print(len(np.where(a == -1)[0]))
            accuracy[0, c] = 140 - len(np.where(a == -1)[0])
            a = clf.predict(np.row_stack((x_t[0:c * 140, :], x_t[c * 140 + 140:, :])))
            print(len(np.where(a == -1)[0]))
            accuracy[1, c] = len(np.where(a == -1)[0])
            np.savetxt(address + 'accurac1class.csv', accuracy, delimiter=',')

    def randomforest(self, X, labels_true_for_Test, x_test, y_test):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from scipy.stats import spearmanr, pearsonr

        rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=15)

        rf.fit(X, labels_true_for_Test)
        predicted_test = rf.predict(x_test)
        predicted_binary = np.argmax(predicted_test, axis=1)
        predic = np.zeros_like(predicted_test)
        for i in range(0, len(predicted_binary)):
            predic[i, int(predicted_binary[i])] = 1
        test_score = r2_score(y_test, predicted_test)
        accuracy = 0
        for i in range(0, len(y_test)):
            if y_test[i, int(predicted_binary[i])] == 1:
                accuracy += 1
        print(accuracy / len(np.where(y_test != 0)[0]))

    def neuralnet_for_classification(self, x_train, y_train, x_test, y_test, features, alfa, dim, dim2, address, nepoch):

        # network model already available
        model = keras.models.load_model(address + 'myNetunsupervised.h5')
        y_pred_val = model.predict(x_test)
        accuracy = ((np.shape(np.nonzero(np.argmax(y_pred_val,axis=1) - np.argmax(y_test,axis=1))))[1])/np.shape(x_test)[0]
        print('neural_netacc = ' + str(accuracy))

    def DEC_test_and_newspeciescomputation(self, x_testT, files, address):
        predicted = np.zeros((140, 10, 10))
        MOC = np.zeros((10, 1))
        MOC2 = np.zeros((10, 1))

        import copy

        for j in range(0, 10):
            out_of_class = 0
            model = keras.models.load_model(address + '/TRAINED DETECTORS/' + files[j] + '.h5')
            x_testz = copy.copy(x_testT)
            x_test2 = self.normalize_test_train_for_newclasses(131, x_testz, files[j])
            count = 0
            anom = 0
            for species in range(0, 10):
                if j != species:
                    out_of_class += np.sum(np.argmax(model.predict(x_test2[species * 140:species * 140 + 140, :]), axis=1))
                    predicted[:, species, j] = np.argmax(model.predict(x_test2[species * 140:species * 140 + 140, :]),
                                                         axis=1)
                    count += 1
                    anom += np.sum((np.argmax(model.predict(x_test2[species * 140:species * 140 + 140, :]), axis=1)))
                if j == species:
                    classin = np.sum(np.argmax(model.predict(x_test2[species * 140:species * 140 + 140, :]), axis=1))
                    print('classification accuracy' + files[j] + '=' + str(140 - classin))
            print('anomaly detection accuracy' + files[j] + '=' + str(anom))
            in_class = np.shape(np.argmax(model.predict(x_test2[j * 140:j * 140 + 140, :]), axis=1))
            MOC[j] = out_of_class
            MOC2[j] = in_class

        print(np.sum(np.prod(predicted[:, 0, 1:], axis=1)))

        for j in range(1, 9):
            print(np.sum(np.prod(np.column_stack((predicted[:, j, 0:j], predicted[:, j, j + 1:])), axis=1)))

        print(np.sum(np.prod(predicted[:, 9, :-1], axis=1)))

    def __init__(self, address, image_segmentation_processing=0, feature_recomputing=0, unsupervised_partitioning=1, classification=1,
                 DEC_testing=1, oneclassSVM=0, robustness_test=0):

        if 'LENSLESS' in address:
            self.static = 0
        else:
            self.static = 1

        if self.static==1:
            self.static = 1
            self.dim = 90
            self.dim_Train = 75
            DEC_testing = 0
            oneclassSVM = 0

        else:
            self.static = 0
            self.dim = 500
            self.dim_Train = self.dim


        try:
            self.files = os.listdir(address + 'TRAIN_FEATURES/')

            for i in range(0, len(self.files)):
                self.files[i] = self.files[i].upper()


        except:
            self.files = os.listdir(address + 'TRAIN_IMAGE/')

        self.address = address
        self.alfa = len(self.files)
        # size of training data


        # size of testing data
        self.dim2 = 140

        # we read the files resulting from the IMAGE PROCESSOR FOR THE PLANKTON CLASSIFIER
        #

        if image_segmentation_processing == 1:
            self.image_processor(address)

        if feature_recomputing == 1:
            self.feature_extractor(address=address, train_folder=address + 'TRAIN_IMAGE/',
                                   train_folder_bin=address + 'BIN_TRAIN_IMAGE/', files=self.files)

        x_train, y_train, x_test, y_test, labels = self.reading(address)

        self.X3 = copy.copy(x_train)
        self.X4 = copy.copy(x_train)
        self.X5 = copy.copy(x_train)
        self.X6 = copy.copy(x_train)

        # number of extracted features
        self.features = x_train.shape[1]
        # let us visualize our space using the PCA

        if unsupervised_partitioning == 1:
            # self.class_imbalance_test(self.address + 'TRAIN_FEATURES\\',y_train)
            # unsupervised partitioning of plankton classifier, returning our labels
            X, components = self.PCA_custom(x_train, y_train)
            self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
            self.x_test_NN = copy.copy(x_test)
            self.x_test_DEC = copy.copy(x_test)
            #### mixture of gaussians
            # gmm = GMM(X2,X,y_train)

        if classification == 1:
            if unsupervised_partitioning == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            ### classification with neural network

            self.X4[np.isnan(self.X4)] = 0

            self.X4[np.isinf(self.X4)] = 0

            self.X4[np.isnan(self.X4)] = 0

            self.X4[np.isinf(self.X4)] = 0

            from sklearn import model_selection




            if self.static==1:
                dim_Train = self.dim_Train
                X_train = np.zeros((self.dim_Train * self.alfa, 131))
                # Y_train = np.zeros(( self.dim_Train*self.alfa,self.alfa))

                # X_train = self.X6

                y_train = self.y_train4

                dim_Test = self.dim - self.dim_Train
                X_test = np.zeros(((dim_Test * self.alfa), 131))
                Y_test = np.zeros(((dim_Test * self.alfa), np.shape(y_train)[1]))

                for i in range(0, np.shape(y_train)[1]):
                    # dim_Train = int(num_*0.7)

                    X_train[i * dim_Train:i * dim_Train + dim_Train, :] = self.X4[i * self.dim:i * self.dim + dim_Train, :]
                    # Y_train[i * dim_Train:i * dim_Train + dim_Train, :] =     y_train[i*self.dim:i*self.dim+dim_Train,:]

                    X_test[i * dim_Test:i * dim_Test + dim_Test, :] = self.X4[
                                                                      i * self.dim + self.dim_Train:i * self.dim + self.dim, :]
                    Y_test[i * dim_Test:i * dim_Test + dim_Test, :] = y_train[
                                                                      i * self.dim + self.dim_Train:i * self.dim + self.dim, :]

                X_train, X_test = self.normalize_test_train(self.features, X_train, X_test)

                X_train = np.column_stack((X_train[:, 0:118], X_train[:, 122:]))
                X_test = np.column_stack((X_test[:, 0:118], X_test[:, 122:]))

                self.randomforest(X_train, self.labels_true_for_Test, X_test, Y_test)

            else:
                X_train, X_test = self.normalize_test_train(self.features, x_train, x_test)
                self.neuralnet_for_classification(X_train, self.labels_true_for_Test, X_test, y_test, self.features,
                                                  self.alfa, self.dim,
                                                  self.dim2, address, 250)
            ##########classification with random forest

        elif classification == 2:
            if unsupervised_partitioning == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.X3, x_test_NN = self.normalize_test_train(self.features, self.X3, self.x_test_NN)
            self.randomforest(self.X3, self.labels_true_for_Test, self.x_test_NN, y_test)

        if DEC_testing == 1:
            # anomaly detector of plankton classifier
            if unsupervised_partitioning == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.DEC_test_and_newspeciescomputation(self.x_test_DEC, self.files, address)

        if oneclassSVM == 1:
            # anomaly detector based on one class SVM
            if unsupervised_partitioning == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.oneclassSVM(self.X5, self.labels_true_for_Test, self.x_test_DEC, y_test, address)
        if robustness_test == 1:
            self.robustness_test(self.X5, y_train, self.features, address, self.files)

####### Insert here the address for the PLANKTON DATASET
address = 'C:\\Users\\VitoPaoloPastore\\Desktop\\DATA\\WHOI DATASET\\' #### change this line with actual address

Test = PLANKTON_CLASSIFIER(address=address, image_segmentation_processing=0, feature_recomputing=0,
                           unsupervised_partitioning=1, classification=1, DEC_testing=1, oneclassSVM=1)






