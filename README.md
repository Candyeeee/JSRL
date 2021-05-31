# JSRL
This is a code implemention of the JSRL method proposed in the manuscipt "Assessing Clinical Progression from Subjective Cognitive Decline to Mild Cognitive Impairment with Incomplete Multi-modal Neuroimages".
# Code Description
Prerequisites used in our code (This is a reference. You also can use different versions of these prerequisetes.)  
python == 3.6  
tf == 1.15.0  
scikit-image 0.17.2  
scipy == 1.5.4  
numpy == 1.19.4  
code composition  
main.py——including the training, test and eval for image synthesis functions  
model.py——defining the network structures used in the JSRL  
layer.py——defining the different network layers, such as convolutional layer, deconvolutional layer and so on  
evaluate.py——evaluating the classification performance of the trained model
# Dataset
Two datasets, including a publicly available Alzheimer’s Disease Neuroimaging Initiative database (ADNI) dataset and a private dataset from the Chinese Longitudinal Aging Study (CLAS) are used in our work.
You can download the ADNI dataset via this link:adni.loni.usc.edu
