# Code for "Annotation-free learning of plankton for classification and anomaly detection"
Vito Paolo Pastore, Thomas Zimmerman, Sujoy K. Biswas and Simone Bianco. 

Plankton classifier.py contains the code to implement the Plankton Classifier pipeline.


The code is organized as a class (Plankton classifier). The main class allows to perform test and evaluate the results, as discussed in the paper,
for both the lensless microscope (or in general, video data) or the WHOI dataset (static data).

The data is accessible at:
https://ibm.box.com/s/8g2mp5knl2by7cv0ie0fx60mlb3rs6v3

Data folder contains both the lensless and WHOI dataset. The datasets contain a 'TRAINING_IMAGE', 'BIN_TRAINING IMAGE' and a
'TRAINING_FEATURES' folders, same for test. Lensless dataset contain also the folder 'TRAINED DETECTORS' with the DEC-detectors models saved in format 'h5' together with the model of a trained neural network for classification (for testing as reported in the paper). 

For details about the total number of image per class and correspondent dataset, please refer to the paper. 
'BIN_TRAIN_IMAGE'
Segmentation resulting image
'TRAIN_FEATURES'
Features extracted. 

The initialization module to instantiate the class and performing the tests is:

  Test = PLANKTON_CLASSIFIER(address=address, image_segmentation_processing=0, feature_recomputing=0,
                           unsupervised_partitioning=1, classification=1, DEC_testing=1, oneclassSVM=1)
                          

Where address is the folder DATA downloaded from the provided links.

Image_segmentation_processing: if ==1 -> calls the image_processor module and performs image segmentation (creates folder 'BIN_TRAIN_IMAGE')

feature_recomputing: if == 1 -> calls the features extraction module and performs feature extraction (creates folder 'TRAIN_FEATURES')

Unsupervised_partitioning:  if == 1 -> calls the unsupervised partitioning module and performs the unsupervised partitioning based on Fuzzy K-means

Classification: if ==1 -> classification based on Neural network on test data, if ==2 ->  classification based on Random Forest on test data

DEC_testing: if ==1 -> DEC detectors test on single classes, and one-out test for detection of new species 

oneclassSVM: if ==1 -> One class SVM test for anomaly detection. 

In order to run the tests and replicate the results described in the manuscript, it is necessary to:
install python and the following set of packages:

Recommended requirements: python 3.6.0
Keras 2.2.4
Tensorflow-gpu 1.9.0
openCV 3.4.1 
scipy 1.1.0
sklearn 0.20.0
numpy 1.15.3
h5py 2.8.0

Substitute the address at line (1329) with the address of the downloaded dataset in your PC.
Choose and set the modules as explained before.
Run the module. 


In the next section, we will describe all the included modules:


normalize_test_train_for_newclasses -> normalization needed for test set with respect to the training data
class local binary patterns -> implementation for local binary patterns 
adjust_gamma -> gamma correction for image
image_processor -> segmentation module. 
feature_extractor -> Feature extraction module
euclidian_distance -> euclidian distance between two arrays
normalize_test_train -> normalization needed for test set with respect to the training data using both test and train data. 
normalize -> normalize between 0 and 1. 
reading -> code for reading the data
evaluate_purity -> customized version of the classic definition of purity 
evaluate_purity_OVERLAP -> revised version of the classic definition of purity 
isdifferente -> number of repetitions in array
PCA_custom -> performs the PCA analysis
clusters_comp -> code for computing clusters
unsupervised_partitioning -> clustering module 
GMM -> mixture of gaussians customized algorithm 
oneclassSVM -> one class SVM algorith 
randomforest -> random forest based classification 
neuralnet_for_classification -> neural net based classification 
DEC_test_and_newspeciescomputation -> all the test using the DEC detectors as described into the paper. 



