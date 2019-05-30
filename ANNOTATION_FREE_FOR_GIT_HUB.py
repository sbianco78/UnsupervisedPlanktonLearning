import matplotlib
matplotlib.use('Agg')
from skimage import feature
from itertools import cycle
from sklearn.metrics import confusion_matrix
from keras import Model
from scipy import interp
from keras.optimizers import Adam
import cv2
import keras
from mahotas.zernike import zernike_moments
from mahotas.features import haralick
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


class ANNOTATION_FREE_PAPER_CODE():

    def normalize_test_train_for_newclasses(self, features, x_test, name):

        medie_per_norm = np.loadtxt('/Users/vitopaolopastore/Desktop/TRAINED DETECTORS/medie' + name + '.csv')
        medie_per_std = np.loadtxt('/Users/vitopaolopastore/Desktop/TRAINED DETECTORS/medie1' + name + '.csv')

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
    def image_processor(self):

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

                for aa in range(0, len(files2)):

                    if files2[aa] != ".DS_Store":

                        img = cv2.imread(train_folder + '/' + files[j] + '/' + files2[aa])
                        img = cv2.medianBlur(img, 3, img)

                        try:

                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # img = adjust_gamma(img, gamma=2.0)
                            plt.imshow(img)


                        except:
                            pass

                        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        img = cv2.bitwise_not(img)
                        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
                        plt.imshow(img)

                        img = cv2.convertScaleAbs(img)
                        im3 = copy.copy(img)
                        im3 = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)
                        # edge detection

                        im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        z = 0
                        frame4 = img

                        ar_max = 0
                        ar_max2 = 0
                        for i in x:
                            # minimum rectangle containing the object
                            PO2 = cv2.boundingRect(i)
                            area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                            # LET'S CHOOSE THE AREA THAT WE WANT
                            if area > 400 and area < 200000:

                                moments = cv2.moments(i)
                                ar = moments['m00']
                                if ar > ar_max2:
                                    ar_max = i
                                    ar_max2 = ar

                        try:

                            i = ar_max
                            PO2 = cv2.boundingRect(i)
                            area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                            cv2.drawContours(im3, [i], -1, (0, 0, 255), -1)
                            #######################display objects###########################ll
                            cv2.imwrite(output_segmentation_train + '/' + files[j] + '/' + files2[aa], im3)
                        except:
                            pass

    def feature_extractor(self, address, train_folder, train_folder_bin, files):

        train_folder = address + 'TRAIN_IMAGE/'
        train_folder_bin = address + 'TRAIN_IMAGEBIN/'
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

                files2 = os.listdir(train_folder_bin  + files[j])
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
                            uuu[:, :, 0] = cv2.bitwise_and(uuu[:, :, 0], mask)
                            uuu[:, :, 1] = cv2.bitwise_and(uuu[:, :, 1], mask)
                            uuu[:, :, 2] = cv2.bitwise_and(uuu[:, :, 2], mask)

                            momentsintensity[5] = np.mean(uuu[:, :, 0]) / np.mean(uuu[:, :, 2])
                            momentsintensity[6] = np.mean(uuu[:, :, 0]) / np.mean(uuu[:, :, 1])
                            momentsintensity[7] = np.mean(uuu[:, :, 1]) / np.mean(uuu[:, :, 2])
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
                                if area > 100 and area < 200000:

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
                            if area > 300 and area < 200000:

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
                                    np.savetxt(train_features_bin_OUTPUT + '\\' + files[j] + '\\' + files2[conteggio] + '.csv',
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
    def normalize_test_train(self,features, X, x_test):

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

    def DEC_Detector(self, x_train, y_train, x_test, y_test, features, M, K, X2, model_name, A, spec,labels_true_for_Test):

        # network definition DEC detector

        NB_CLASSES = 2
        VERBOSE = 1
        BATCH_SIZE = 150
        NEPOCH = 200
        sgd = Adam(lr=0.0000075)
        s = np.random.randint(0, A, K)

        roc_val2 = np.zeros((NEPOCH, 2), dtype=np.float)

        class roc_callback(Callback):

            def __init__(self, training_data, validation_data):
                self.x = training_data[0]
                self.y = training_data[1]
                self.x_val = validation_data[0]
                self.y_val = validation_data[1]
                self.epoc = 0

            def inc_epoc(self, logs={}):
                self.epoc += 1

            def on_train_begin(self, logs={}):
                return

            def on_train_end(self, logs={}):
                return

            def on_epoch_begin(self, epoch, logs={}):
                return

            def on_epoch_end(self, epoch, logs={}):

                # on epoch end, we have to shuffle again the anomaly training set, for more generalization
                y_pred_val = self.model.predict(self.x_val)

                y_pred = self.model.predict(self.x)

                num_species = self.alfa
                M4 = np.zeros((1, num_species), dtype=np.int)
                M4[0, 0] = len(X2[np.where(labels_true_for_Test[:, 0] == 1)])

                for asd in range(1, num_species):
                    M4[0, asd] = len(X2[np.where(labels_true_for_Test[:, asd] == 1)])

                DIM5 = int(M4[0, spec] / (self.alfa - 1))
                M2 = len(X2[np.where(labels_true_for_Test[:, spec] == 1)])

                PPP = self.x[0:A, :]

                if spec != self.alfa - 1:
                    for j in range(0, self.alfa):

                        if j != spec:
                            s1 = np.random.uniform(0, M4[0, j], DIM5 + 100).astype(int)

                            if j != self.alfa - 1:
                                PPP = np.row_stack((PPP, X2[np.where(labels_true_for_Test[:, j] == 1)][
                                    s1[0:int(M2 / (self.alfa - 1))]]))
                            else:
                                PPP = np.row_stack((PPP, X2[np.where(labels_true_for_Test[:, j] == 1)][
                                                         s1[0: M2 - (self.alfa - 2) * int(M2 / (self.alfa - 1))], :]))

                    self.x = PPP
                else:
                    for j in range(0, self.alfa):

                        if j != spec:
                            s1 = np.random.uniform(0, M4[0, j], DIM5 + 100).astype(int)

                            if j != 0:
                                PPP = np.row_stack(
                                    (PPP,
                                     X2[np.where(labels_true_for_Test[:, j] == 1)][s1[0:int(M2 / (self.alfa - 1))]]))
                            else:
                                PPP = np.row_stack((PPP, X2[np.where(labels_true_for_Test[:, j] == 1)][
                                                         s1[0: M2 - (self.alfa - 2) * int(M2 / (self.alfa - 1))], :]))

                    self.x = PPP

                global fpr
                global tpr
                global roc_auc

                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                itest = np.zeros((self.alfa * self.dim2, NB_CLASSES), dtype=np.int)
                intest = np.zeros((self.alfa * self.dim2, NB_CLASSES), dtype=np.int)

                for i in range(self.alfa * self.dim2):
                    bin = np.argmax(y_pred_val[i, :])

                    itest[i, bin] = 1

                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(itest[:, i], self.y_val[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(itest.ravel(), self.y_val.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NB_CLASSES)]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(NB_CLASSES):
                    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                # Finally average it and compute AUC
                mean_tpr /= NB_CLASSES
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                score = model.evaluate(self.x_val, self.y_val, batch_size=BATCH_SIZE, verbose=VERBOSE)
                roc = roc_auc_score(self.y, y_pred)
                j = np.argmax(y_pred_val, axis=0)
                roc_val = roc_auc_score(self.y_val, y_pred_val)
                print('roc-auc: %s - roc-auc_val: %s' % (str(round(roc, 2)), str(round(roc_val, 2))),
                      end=100 * ' ' + '\n')
                print('test_score' + str(score))
                y1 = np.argmax(self.y_val, axis=1)
                y2 = np.argmax(y_pred_val, axis=1)
                matrix = confusion_matrix(y1, y2)
                print(matrix)
                roc_val2[int(self.epoc), 0] = roc
                roc_val2[int(self.epoc), 1] = roc_val
                mattt = np.zeros((2, 2), dtype=float)
                self.inc_epoc()

                if self.epoc == NEPOCH:
                    # this part of the code is used for matrix of confusion saving
                    np.savetxt(address + model_name + 'MOC.csv', matrix)

                    # this part of the code is used for matrix of confusion visualization

                    mattt[0, 0] = float(matrix[0, 0]) / self.dim2
                    mattt[0, 1] = float(matrix[0, 1]) / self.dim2
                    mattt[1, 0] = float(matrix[1, 0]) / (self.dim2 * (self.alfa - 1))
                    mattt[1, 1] = float(matrix[1, 1]) / (self.dim2 * (self.alfa - 1))
                    labels = [model_name, 'anomaly']
                    fig = plt.figure(figsize=(10, 9))
                    plt.rcParams['image.cmap'] = 'gray'
                    s = plt.imshow(mattt)
                    plt.xticks(np.arange(2), labels, size=12, rotation=90)
                    plt.yticks(np.arange(2), labels, size=12, rotation=0)
                    colorbar_ax = fig.add_axes()
                    font_size = 14  # Adjust as appropriate.
                    fig.subplots_adjust(bottom=0.52)
                    fig.subplots_adjust(left=0.1)
                    fig.colorbar(s, cax=colorbar_ax).ax.tick_params(labelsize=font_size)
                    plt.savefig(address + model_name + 'matrix')
                    # we want to select best accuracy over training
                    np.savetxt(address + 'score.csv', score, delimiter=",")

                    # plt.show()
                return

            def on_batch_begin(self, batch, logs={}):
                return

            def on_batch_end(self, batch, logs={}):
                return

        # code to be used for training
        want_retrain = 1

        if want_retrain == 1:
            model = create_model(M, K, x_train, features, s)

            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.summary()
            history = model.fit(x_train, y_train, epochs=NEPOCH, batch_size=BATCH_SIZE, validation_split=0.1,
                                verbose=VERBOSE,
                                callbacks=[roc_callback(training_data=(x_train, y_train),
                                                        validation_data=(x_test, y_test))])  # callbacks= [forcall])

            np.savetxt(address + 'matrixunsupervised.csv', roc_val2, delimiter=",")
            model.save(address + model_name + '.h5', overwrite=True)
            print(history.history.keys())
            plt.figure(figsize=(10, 9))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=5)

            lw = 2
            labels = [model_name, 'anomaly']
            colors = cycle(['cornflowerblue', 'darkorange', 'black', 'red', 'black'])
            for i, color in zip(range(NB_CLASSES), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label=labels[i] + ' (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.ylabel('True Positive Rate', fontsize=20)
            plt.legend(loc="lower right", fontsize=15)
            plt.title('ROC curve', size=20)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.savefig('/Users/vitopaolopastore/Desktop/' + model_name + 'ROC')

    # DEC architecture model creation
    def create_model(self, M, x_train, features, s):

        inp1 = Input(shape=(features,))
        outoriginal = Dense(M, input_shape=(features,), activation='relu')(inp1)
        out0 = Lambda(lambda x: np.abs(x_train[int(s[0])] - x))(inp1)
        out0 = Dense(M, input_shape=(features,), activation='relu')(out0)
        out1 = Lambda(lambda x: np.abs(x_train[int(s[1])] - x))(inp1)
        out1 = Dense(M, input_shape=(features,), activation='relu')(out1)
        out2 = Lambda(lambda x: np.abs(x_train[int(s[2])] - x))(inp1)
        out2 = Dense(M, input_shape=(features,), activation='relu')(out2)
        out3 = Lambda(lambda x: np.abs(x_train[int(s[3])] - x))(inp1)
        out3 = Dense(M, input_shape=(features,), activation='relu')(out3)
        out4 = Lambda(lambda x: np.abs(x_train[int(s[4])] - x))(inp1)
        out4 = Dense(M, input_shape=(features,), activation='relu')(out4)
        out5 = Lambda(lambda x: np.abs(x_train[int(s[5])] - x))(inp1)
        out5 = Dense(M, input_shape=(features,), activation='relu')(out5)
        out6 = Lambda(lambda x: np.abs(x_train[int(s[6])] - x))(inp1)
        out6 = Dense(M, input_shape=(features,), activation='relu')(out6)
        out7 = Lambda(lambda x: np.abs(x_train[int(s[7])] - x))(inp1)
        out7 = Dense(M, input_shape=(features,), activation='relu')(out7)
        out8 = Lambda(lambda x: np.abs(x_train[int(s[8])] - x))(inp1)
        out8 = Dense(M, input_shape=(features,), activation='relu')(out8)
        out9 = Lambda(lambda x: np.abs(x_train[int(s[9])] - x))(inp1)
        out9 = Dense(M, input_shape=(features,), activation='relu')(out9)
        out10 = Lambda(lambda x: np.abs(x_train[int(s[10])] - x))(inp1)
        out10 = Dense(M, input_shape=(features,), activation='relu')(out10)
        out11 = Lambda(lambda x: np.abs(x_train[int(s[11])] - x))(inp1)
        out11 = Dense(M, input_shape=(features,), activation='relu')(out11)
        out12 = Lambda(lambda x: np.abs(x_train[int(s[12])] - x))(inp1)
        out12 = Dense(M, input_shape=(features,), activation='relu')(out12)
        out13 = Lambda(lambda x: np.abs(x_train[int(s[13])] - x))(inp1)
        out13 = Dense(M, input_shape=(features,), activation='relu')(out13)
        out14 = Lambda(lambda x: np.abs(x_train[int(s[13])] - x))(inp1)
        out14 = Dense(M, input_shape=(features,), activation='relu')(out14)
        out15 = Lambda(lambda x: np.abs(x_train[int(s[15])] - x))(inp1)
        out15 = Dense(M, input_shape=(features,), activation='relu')(out15)
        out16 = Lambda(lambda x: np.abs(x_train[int(s[16])] - x))(inp1)
        out16 = Dense(M, input_shape=(features,), activation='relu')(out16)
        out17 = Lambda(lambda x: np.abs(x_train[int(s[17])] - x))(inp1)
        out17 = Dense(M, input_shape=(features,), activation='relu')(out17)
        out18 = Lambda(lambda x: np.abs(x_train[int(s[18])] - x))(inp1)
        out18 = Dense(M, input_shape=(features,), activation='relu')(out18)
        out19 = Lambda(lambda x: np.abs(x_train[int(s[19])] - x))(inp1)
        out19 = Dense(M, input_shape=(features,), activation='relu')(out19)
        out20 = Lambda(lambda x: np.abs(x_train[int(s[20])] - x))(inp1)
        out20 = Dense(M, input_shape=(features,), activation='relu')(out20)
        out21 = Lambda(lambda x: np.abs(x_train[int(s[21])] - x))(inp1)
        out21 = Dense(M, input_shape=(features,), activation='relu')(out21)
        out22 = Lambda(lambda x: np.abs(x_train[int(s[22])] - x))(inp1)
        out22 = Dense(M, input_shape=(features,), activation='relu')(out22)
        out23 = Lambda(lambda x: np.abs(x_train[int(s[23])] - x))(inp1)
        out23 = Dense(M, input_shape=(features,), activation='relu')(out23)
        out24 = Lambda(lambda x: np.abs(x_train[int(s[24])] - x))(inp1)
        out24 = Dense(M, input_shape=(features,), activation='relu')(out24)

        merged_vector = concatenate(
            [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
             out16, out17, out18, out19, out20, out21, out22, out23, out24], axis=-1)

        out_concatenation = (Dense(M, input_shape=(M * 25,), activation='relu'))(merged_vector)
        conc_2_net = concatenate([out_concatenation, outoriginal], axis=-1)

        out_concatenation = Dense(10, activation='relu', name='layer3')(conc_2_net)
        fine = Dense(2, activation='softmax', name='layer4')(out_concatenation)
        model = Model(inputs=[inp1], outputs=[fine])
        ######## end model
        return model

    def reading(self, address):
        train_folder = address + 'TRAIN_FEATURES/'
        mydir = train_folder
        test_folder = address + 'TEST_FEATURES/'
        files = ['ARCELLA VULGARIS', 'ACTINOSPHAERIUM NUCLEOFILUM', 'STENTOR COERULEUS', 'DIDINIUM NASUTUM',
                 'EUPLOTES EURYSTOMUS', 'SPIROSTOMUM AMBIGUUM', 'BLEPHARISMA AMERICANUM', 'VOLVOX', 'DILEPTUS',
                 'PARAMECIUM  BURSARIA']
        labels = list()
        mydir2 = test_folder
        files3 = os.listdir(mydir2)

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

                # flag for reading
                aa = 0

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

        import copy

        y_train = y_train[0:self.alfa * self.dim, 0:self.alfa]
        y_test = y_test[0:self.alfa * self.dim2, 0:self.alfa]

        np.savetxt(address + 'x_train_for_further_code.csv', x_train,
                   delimiter=",")
        np.savetxt('/Users/vitopaolopastore/Desktop/' + 'x_test_for_further_code.csv', x_test,
                   delimiter=",")

        return x_train, y_train, x_test, y_test, labels

    def features_importance(self, x_train, y_train):
        # code for determining features ranking
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel
        clf = ExtraTreesClassifier(n_estimators=100, )
        clf = clf.fit(x_train, y_train)
        importance = clf.feature_importances_
        return importance

    def evaluate_purity(self, k, predicted, Y, size):

        score_best = np.zeros((k, k))

        for i in range(0, k):
            for j in range(0, k):
                # computing all the possible overlap
                score_best[i, j] = np.count_nonzero(Y[predicted[i], j] == 1)

        # taking the maximum arguments and values for accuracy
        master = np.argmax(score_best, axis=1)
        maxs = np.max(score_best, axis=1)
        purit = np.sum(maxs) / size
        return master, purit

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
        from sklearn import preprocessing
        from sklearn.decomposition import PCA

        std_scale = preprocessing.StandardScaler().fit(x_def)
        x_def = std_scale.transform(x_def)

        # PCA from scikit-learn
        pca = PCA(n_components=self.features)

        pca.fit(x_def)
        X = pca.transform(x_def)
        files = ['ARCELLA VULGARIS', 'ACTINOSPHAERIUM NUCLEOFILUM', 'STENTOR COERULEUS', 'DIDINIUM NASUTUM',
                 'EUPLOTES EURYSTOMUS', 'SPIROSTOMUM AMBIGUUM', 'BLEPHARISMA AMERICANUM', 'VOLVOX', 'DILEPTUS',
                 'PARAMECIUM  BURSARIA']
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
        ax.legend()

        plt.show()

        return X, eigenvectors

    def unsupervised_partitioning(self, x_train, y_train,X):
        k = self.alfa
        X2 = copy.copy(x_train)
        X2 = self.normalize(X2)
        fuzzy_kmeans = FuzzyKMeans(k=self.alfa, m=1.6)
        fuzzy_kmeans.fit(X2)
        fig2 = plt.figure()
        num_species = self.alfa
        ax = fig2.add_subplot(111, projection='3d')
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, self.alfa))
        colorss = colors
        labels = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7', 'cluster 8',
                  'cluster 9', 'cluster 10', 'cluster11', 'cluster12', 'cluster13', 'cluster14', 'cluster15', 'cluster 16',
                  'cluster 17', 'cluster 18', 'cluster19', 'cluster20', 'cluster21', 'cluster22']
        predicted = list()
        for p in range(0, k):
            predicted.append(list())
        for i in range(0, k):
            predicted[i].append(np.where(fuzzy_kmeans.labels_ == i))
        master, purit = self.evaluate_purity(self.alfa, predicted, y_train, self.alfa * self.dim)
        for i in range(0, k):
            ax.scatter(X[predicted[i], 0], X[predicted[i], 1], X[predicted[i], 2], c=colorss[master[i]], edgecolor='k',
                       label=labels[master[i]])
        plt.legend()
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

        labels_true_for_Test = np.zeros((self.alfa * self.dim, k))

        for aaa in range(0, self.alfa):
            labels_true_for_Test[predicted[aaa], master[aaa]] = 1

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
        master, purit = self.evaluate_purity(k, predicted1, y_train, 5000)

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

    def oneclassSVM(self,X2, y_train, x_test, y_test,address):

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

        print(accuracy/1400)

    def neuralnet_for_classification(self, x_train, y_train, x_test, y_test, features,alfa,dim,dim2,address,nepoch):
        # network definition
        NB_CLASSES = self.alfa
        VERBOSE = 1
        BATCH_SIZE = 50
        NEPOCH = nepoch
        sgd = RMSprop(0.0001)
        model = Sequential()
        model.add(Dense(40, input_shape=(features,), kernel_regularizer=l1(0.005)))

        model.add(BatchNormalization())

        model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(40, input_shape=(features,), kernel_regularizer=l1(0.005)))  # ,)

        model.add(BatchNormalization())

        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NB_CLASSES))
        model.add(Activation('softmax'))
        roc_val2 = np.zeros((NEPOCH, self.alfa), dtype=np.float)

        ######## end model

        class roc_callback(Callback):
            def __init__(self, training_data, validation_data):
                self.x = training_data[0]
                self.y = training_data[1]
                self.x_val = validation_data[0]
                self.y_val = validation_data[1]
                self.epoc = 0
                self.files = ['ARCELLA VULGARIS', 'ACTINOSPHAERIUM NUCLEOFILUM', 'STENTOR COERULEUS', 'DIDINIUM NASUTUM',
                         'EUPLOTES EURYSTOMUS', 'SPIROSTOMUM AMBIGUUM', 'BLEPHARISMA AMERICANUM', 'VOLVOX', 'DILEPTUS',
                         'PARAMECIUM  BURSARIA']

            def inc_epoc(self, logs={}):
                self.epoc += 1

            def on_train_begin(self, logs={}):
                return

            def on_train_end(self, logs={}):

                return

            def on_epoch_begin(self, epoch, logs={}):
                return

            def on_epoch_end(self, epoch, logs={}):

                y_pred_val = self.model.predict(self.x_val)

                y_pred = self.model.predict(self.x)

                global fpr
                global tpr
                global roc_auc

                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                itest = np.zeros((alfa * dim2, alfa), dtype=np.int)
                intest = np.zeros((alfa * dim2, alfa), dtype=np.int)

                for i in range(alfa * dim2):
                    bin = np.argmax(y_pred_val[i, :])

                    itest[i, bin] = 1

                for i in range(NB_CLASSES):
                    fpr[i], tpr[i], _ = roc_curve(itest[:, i], self.y_val[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(itest.ravel(), self.y_val.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NB_CLASSES)]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(NB_CLASSES):
                    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                # Finally average it and compute AUC
                mean_tpr /= NB_CLASSES
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                score = model.evaluate(self.x_val, self.y_val, batch_size=BATCH_SIZE, verbose=VERBOSE)
                roc = roc_auc_score(self.y, y_pred)
                j = np.argmax(y_pred_val, axis=0)
                roc_val = roc_auc_score(self.y_val, y_pred_val)
                print('roc-auc: %s - roc-auc_val: %s' % (str(round(roc, 10)), str(round(roc_val, 10))),
                      end=100 * ' ' + '\n')
                print('test_score' + str(score))
                y1 = np.argmax(self.y_val, axis=1)
                y2 = np.argmax(y_pred_val, axis=1)
                matrix = confusion_matrix(y1, y2)
                print(matrix)
                roc_val2[int(self.epoc), 0] = roc
                roc_val2[int(self.epoc), 1] = roc_val

                self.inc_epoc()

                if self.epoc == NEPOCH:
                    matrix = matrix / np.max(matrix)
                    labels = self.files
                    fig = plt.figure()
                    plt.rcParams['image.cmap'] = 'inferno'
                    s = plt.imshow(matrix)
                    plt.xticks(np.arange(10), labels, size=12, rotation=90)
                    plt.yticks(np.arange(10), labels, size=12, rotation=0)
                    colorbar_ax = fig.add_axes()
                    font_size = 14  # Adjust as appropriate.
                    fig.subplots_adjust(bottom=0.52)
                    fig.subplots_adjust(left=0.1)
                    fig.colorbar(s, cax=colorbar_ax).ax.tick_params(labelsize=font_size)
                    plt.show()
                    plt.show()

                return

            def on_batch_begin(self, batch, logs={}):
                return

            def on_batch_end(self, batch, logs={}):
                return

        want_retrain = 1

        if want_retrain == 1:
            # forcall = callback1()
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.summary()
            history = model.fit(x_train, y_train, epochs=NEPOCH, batch_size=BATCH_SIZE, validation_split=0.15,
                                verbose=VERBOSE,
                                callbacks=[roc_callback(training_data=(x_train, y_train),
                                                        validation_data=(x_test, y_test))])  # callbacks= [forcall])

            np.savetxt(address + 'matrixunsupervised.csv', roc_val2, delimiter=",")

            model.save(address + 'myNetunsupervised.h5', overwrite=True)

            print(history.history.keys())

            plt.figure()

            # summarize history for accuracy

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])

            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('/Users/vitopaolopastore/Desktop/Accuracy5.png')
            # summarize history for loss
            plt.figure()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.figure()
            plt.savefig('/Users/vitopaolopastore/Desktop/Accuracy6.png')

            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='Average ROC curve (AUC = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='black', linestyle=':', linewidth=7)

            lw = 2
            labels = self.files
            jet = plt.cm.jet
            colors = jet(np.linspace(0, 1, self.alfa))
            for i, color in zip(range(NB_CLASSES), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label=labels[i] + ' (AUC = {1:0.2f})'
                                           ''.format(i, roc_auc[i]), linewidth=3)

            lw = 2

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR', fontsize=30)
            plt.ylabel('TPR', fontsize=30)
            plt.title('TESTING ROC CURVES', size=30)
            plt.legend(loc="lower right", fontsize=15)
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.tick_params(axis='both', which='minor', labelsize=30)


# code for the training of the several DEC detectors
    def anomaly_base_Test(self):
        score = 0

        for i in range(0, self.alfa):

            x_for_this_loop = copy.copy(X2)

            T = np.row_stack((x_for_this_loop[np.where(labels_true_for_Test[:, i] == 1)]))
            # s = np.random.uniform(0, 1000, 25)
            M2 = len(x_for_this_loop[np.where(labels_true_for_Test[:, i] == 1)])

            if i != 9:
                for j in range(0, self.alfa):

                    if j != i:
                        if j != self.alfa - 1:
                            T = np.row_stack(
                                (T, x_for_this_loop[np.where(labels_true_for_Test[:, j] == 1)][0:int(M2 / 9), :]))
                        else:
                            T = np.row_stack(
                                (T, x_for_this_loop[np.where(labels_true_for_Test[:, j] == 1)][0: M2 - 8 * int(M2 / 9), :]))

                T2 = copy.copy(T)
                x_test_for_this_loop = copy.copy(x_test)

            else:
                for j in range(0, self.alfa):

                    if j != i:
                        if j == 0:

                            T = np.row_stack(
                                (T, x_for_this_loop[np.where(labels_true_for_Test[:, j] == 1)][0: M2 - 8 * int(M2 / 9), :]))
                        else:
                            T = np.row_stack(
                                (T, x_for_this_loop[np.where(labels_true_for_Test[:, j] == 1)][0:int(M2 / 9), :]))

                T2 = copy.copy(T)
                x_test_for_this_loop = copy.copy(x_test)

            Y = np.zeros((2 * M2, 2))
            Y[0:M2, 0] = 1
            Y[M2:2 * M2, 1] = 1

            YTEST = np.zeros((self.alfa * self.dim2, 2))

            try:
                YTEST[i * self.dim2:i * self.dim2 + self.dim2, 0] = 1
            except:
                pass

            for z in range(0, self.alfa * self.dim2):
                if YTEST[z, 0] == 0:
                    YTEST[z, 1] = 1

            model_name = files[i]

            medie_per_norm = np.zeros((features, 1), dtype=np.float)
            medie_per_std = np.zeros((features, 1), dtype=np.float)

            for k in range(0, features):
                medie_per_norm[k] = float(np.max(T[0:M2, k]))
                medie_per_std[k] = float(np.min(T[0:M2, k]))

                x_for_this_loop[:, k] = (x_for_this_loop[:, k] - medie_per_std[k]) / (-medie_per_std[k] + medie_per_norm[k])

            for k in range(0, features):
                x_test_for_this_loop[:, k] = (x_test_for_this_loop[:, k] - medie_per_std[k]) / (
                        -medie_per_std[k] + medie_per_norm[k])

            # FOR THE FIRST TIME IN THE NEURAL NET, WE NEED T (OUR X_tRAIN) TO BE NORMALIZED

            for k in range(0, features):
                T[:, k] = (T[:, k] - medie_per_std[k]) / (-medie_per_std[k] + medie_per_norm[k])

            np.savetxt('/Users/vitopaolopastore/Desktop/medie' + model_name + '.csv', medie_per_norm, delimiter=",")

            TPR = []
            FPR = []
            ROC_AUC = []

            np.savetxt('/Users/vitopaolopastore/Desktop/medie1' + model_name + '.csv', medie_per_std)
            self.DEC_Detector(T, Y, x_test_for_this_loop, YTEST, features, 40, 25, x_for_this_loop, model_name,
                                         M2, i, labels_true_for_Test)
            score_new = np.loadtxt('/Users/vitopaolopastore/Desktop/score.csv', delimiter=",")

            if score_new[1] > score:

                try:
                    os.remove('/Users/vitopaolopastore/Desktop/def' + model_name + '.h5')
                except:
                    pass
                os.rename('/Users/vitopaolopastore/Desktop/' + model_name + '.h5',
                          '/Users/vitopaolopastore/Desktop/def' + model_name + '.h5')
                np.savetxt('/Users/vitopaolopastore/Desktop/bestscore_training.csv', score_new, delimiter=",")
                score = score_new[1]


    def DEC_test_and_newspeciescomputation(self, x_testT, files,address):
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


    def __init__(self, address, image_segmentation_processing=0, feature_recomputing=0, unsupervised=1,classification=1, DEC_testing=1,oneclassSVM = 0, robustness_test = 0, dim=500):
        self.files = ['ARCELLA VULGARIS', 'ACTINOSPHAERIUM NUCLEOFILUM', 'STENTOR COERULEUS', 'DIDINIUM NASUTUM',
                 'EUPLOTES EURYSTOMUS', 'SPIROSTOMUM AMBIGUUM', 'BLEPHARISMA AMERICANUM', 'VOLVOX', 'DILEPTUS',
                 'PARAMECIUM  BURSARIA']

        self.alfa = 10
        # size of training data
        self.dim = dim

        # size of testing data
        self.dim2 = 140

        # we read the files resulting from the IMAGE PROCESSOR FOR THE PLANKTON CLASSIFIER
        x_train, y_train, x_test, y_test, labels = self.reading(address)

        if image_segmentation_processing == 1:
            self.image_processor()

        if feature_recomputing == 1:
            self.feature_extractor(address=address, train_folder=address + 'TRAIN_IMAGE/',
                              train_folder_bin=address + 'BIN_TRAIN_IMAGE/', files=self.files)

        self.X3 = copy.copy(x_train)
        self.X4 = copy.copy(x_train)
        self.X5 = copy.copy(x_train)
        self.X6 = copy.copy(x_train)

        # number of extracted features
        self.features = x_train.shape[1]
        # let us visualize our space using the PCA


        if unsupervised == 1:
            # unsupervised partitioning of plankton classifier, returning our labels
            X, components = self.PCA_custom(x_train, y_train)
            self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train,X)
            self.x_test_NN = copy.copy(x_test)
            self.x_test_DEC = copy.copy(x_test)
            #### mixture of gaussians
            # gmm = GMM(X2,X,y_train)

        if classification == 1:
            if unsupervised == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            #### classification with neural network
            self.X4, self.x_test_NN = self.normalize_test_train(self.features, self.X4, self.x_test_NN)
            self.neuralnet_for_classification(self.X4, self.labels_true_for_Test, self.x_test_NN, y_test, self.features,self.alfa,self.dim,self.dim2,address,100)

            ##########classification with random forest

        elif classification == 2:
            if unsupervised == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.X3, x_test_NN = self.normalize_test_train(self.features, self.X3, self.x_test_NN)
            self.randomforest(self.X3, self.labels_true_for_Test, self.x_test_NN, y_test)


        if DEC_testing == 1:
            # anomaly detector of plankton classifier
            if unsupervised == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.DEC_test_and_newspeciescomputation(self.x_test_DEC, self.files,address)

        if oneclassSVM == 1:
            # anomaly detector based on one class SVM
            if unsupervised == 0:
                X, components = self.PCA_custom(x_train, y_train)
                self.labels_true_for_Test = self.unsupervised_partitioning(x_train, y_train, X)
                self.x_test_NN = copy.copy(x_test)
                self.x_test_DEC = copy.copy(x_test)
            self.oneclassSVM(self.X5,self.labels_true_for_Test,self.x_test_DEC,y_test,address)
        if robustness_test ==1:
            self.robustness_test(self.X5,y_train, self.features,address,self.files)


    def robustness_test(self, x_train, y_train, features,address,files):
        accuracyarr = np.zeros((9, self.alfa))
        medies = np.zeros((10, features))

        for kk in range(0, self.alfa):


            model = keras.models.load_model(address + '/TRAINED DETECTORS/' + files[kk] + '.h5')

            X2 = copy.copy(x_train)

            X4 = copy.copy(x_train)

            maxs = np.loadtxt(address + '\\TRAINED DETECTORS\\medie' + self.files[kk] + '.csv')
            mins = np.loadtxt(address + '\\TRAINED DETECTORS\\medie1' + self.files[kk] + '.csv')

            for k in range(0, features):
                X2[:, k] = (X2[:, k] - mins[k]) / (
                        -mins[k] + maxs[k])

            if kk == 5:
                medies[0, :] = np.mean(X4[kk * 500:kk * 500 + 50, :], axis=0)

            for count in range(0, 9):


                specieinventata = np.zeros((500, features))


                onlytodraw = np.zeros((500, features))

                p = np.zeros((1, self.alfa))
                for assign in range(0, self.alfa):
                    if assign == kk:
                        p[0, assign] = (count + 1) / self.alfa
                    else:
                        p[0, assign] = (1 - (count + 1) / self.alfa) / self.alfa
                for i in range(0, 500):
                    for j in range(0, 131):
                        for k in range(0, self.alfa):
                            specieinventata[i, j] += X2[k * 500 + i, j] * p[0, k]
                            onlytodraw[i, j] += X4[k * 500 + i, j] * p[0, k]

                a = model.predict(specieinventata)
                TP = 0
                for i in range(0, len(a)):
                    if a[i, 0] < a[i, 1]:
                        TP += 1
                accuracyarr[count, kk] = TP / 500
                if kk == 5:
                    medies[count + 1, :] = np.mean(onlytodraw, axis=0)
        data = medies
        targettino = np.zeros((10), dtype=np.int)
        for i in range(0, 10):
            targettino[i] = i
        if draw == 1:

            target_names = self.files
            labels = ['original', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']

            features_name = list()
            # features_name.append('original_' + files[5])
            for i in range(0, 131):
                features_name.append(i)
            from pandas.plotting import parallel_coordinates

            import pandas as pd

            df = pd.DataFrame(data=(np.c_[data]),
                              columns=features_name)
            df['species'] = pd.Categorical.from_codes(targettino, labels)
            plt.figure(figsize=(20, 9))
            parallel_coordinates(df, 'species', color=colors)


Test = ANNOTATION_FREE_PAPER_CODE(address='/Users/vitopaolopastore/Desktop/DATASET_PASTORE/',feature_recomputing=1, unsupervised=0, classification=0, DEC_testing=0, oneclassSVM = 0,robustness_test = 1)







