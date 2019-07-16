# UnsupervisedPlanktonLearning
Plankton classifier.py contains the code to implement the Plankton Classifier pipeline as described in:
"Annotation-free learning of plankton for classification and anomaly detection".

The code is organized as a class (Plankton classifier). The main class allows to perform test and evaluate the results, as discussed in the paper,
for both the lensless microscope (or in general, video data) or the WHOI dataset (static data).

The data is accessible at:
https://ibm.box.com/s/8g2mp5knl2by7cv0ie0fx60mlb3rs6v3

Data folder contains both the lensless and WHOI dataset. The datasets contain a 'TRAINING_IMAGE', 'BIN_TRAINING IMAGE' and a
'TRAINING_FEATURES' folders, same for test. 
For details about the total number of image per class and correspondent dataset, please refer to the paper. 
'BIN_TRAIN_IMAGE'
Segmentation resulting image
'TRAIN_FEATURES'
Features extracted. 

The initialization module for performing the tests is:

  Test = PLANKTON_CLASSIFIER(address=address, image_segmentation_processing=0, feature_recomputing=0,
                           unsupervised_partitioning=1, classification=1, DEC_testing=1, oneclassSVM=1)

Where address is the folder DATA downloaded from the provided links.

Image_segmentation_processing: if ==1 -> calls the image_processor module and performs image segmentation (creates folder 'BIN_TRAIN_IMAGE')
feature_recomputing: if == 1 -> calls the image_processor module and performs feature extraction (creates folder 'TRAIN_FEATURES')
Unsupervised_partitioning:  if == 1 -> calls the image_processor module and performs the unsupervised partitioning based on Fuzzy K-meabs
Classification: if ==1 -> classification based on Neural network on test data, if ==2 ->  classification based on Random Forest on test data
DEC_testing: if ==1 -> DEC detectors test on single classes, and one-out test for detection of new species 
oneclassSVM: if ==1 -> One class SVM test for anomaly detection. 

In the next section, we will describe all the included modules:



