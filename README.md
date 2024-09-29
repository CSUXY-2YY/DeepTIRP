# DeepTIRP
Title: Large-Population based Deep Learning Models in Classifying  Primary Bone Tumors and Bone Infections based on Radiographs
Background: Due to the similarity in image appearance of the lesions from primary bone tumors (PBTs) and bone infections, clinical orthopedic surgeons even professional radiologists may have difficulty distinguishing them. This study aimed to develop an ensemble deep learning algorithm differentiating PBTs from bone infections based on radiographs from multi-center.
Methods: In this retrospective study, 1992 patients from three hospitals with histologically confirmed PBTs or bone infections were identified and included. Patients were divided into a training set (N=1044), a test set (N=354), an internal validation set (N=171) and an external test set (N=423). The ensemble framework was constructed using radiograph images and clinical characteristics for binary classification on PBTs and bone infections. ROC curve and confusion matrix were utilized to evaluate the classification performance, and further compared it with radiologist interpretation with different seniority. Visual analysis was conducted to explain the attention of the models via GradCAM and ScoreCAM.
Results: In the analysis conducted, the ensemble model demonstrated significant performance improvements compared to models relying solely on radiograph images. Specifically, on the internal test set, the ensemble model achieved AUCs of 0.948 (95% CI, 0.931-0.963) and accuracies of 0.881 for binary classification tasks. The consistency of the model on the external test set was demonstrated, with AUCs of 0.963 (95% CI, 0.951-0.973) and accuracies of 0.895. Meanwhile, the performance of the ensemble model was also superior to that of junior radiologist group and medium seniority group (accuracy: 75.8% and 80.2%, respectively), and comparable to that of senior radiologist group (accuracy: 83.6%). The inter-reader reliability (Fleiss κ: 0.800) between models was much higher than that between radiologists (Fleiss κ: 0.401).
Conclusion: Our study built a framework which displayed promising performance and reliable results in classifying PBTs and bone infections. The ensemble model was better than three groups of radiologists with diverse seniority.

# Dependency
```
conda create -n DeepTIRP python=3.8
conda activate DeepTIRP
pip install pandas
pip install numpy
pip install scipy
pip install dominate
pip install torch==1.12.1+cu113
pip install torchvision==0.13.1+cu113
```
### Please Download Pre-trained models for Swin Transformer, Vision Transformer and Efficientnet and place them under the Pretrained_model directory

# Data collection and processing
1.Research participants and data: this retrospective multicenter study collected patients via consecutive sampling between 2013 and 2022 from two cohorts: training cohort (from the Second Xiangya Hospital of Central South University) and testing cohort (from Xiangya Hospital of Central South University and Hunan Children's Hospital of Central South University). These lesions were identified to have bone involvement through pre-operative radiographs and were histologically diagnosed following biopsy or surgery. 
(i) For the inclusion criteria, lesions were confirmed and diagnosed as PBTs according to the 2020 World Health Organization (WHO) system for the classification for tumors of bone 1 while bone infections were confirmed and proven by histology and (or) bacterial culture. The other vital inclusion criteria are evident as well as available clinical information and pre-operative radiographs. 
(ii) The screening criteria: (a) radiographs were from patients diagnosed between 2013 and 2022 (b) in selected three hospitals; (c) radiographs with robust quality for reliable assessments of the bone lesions and (d) all of these radiographs were pre-operative. 
Clinical characteristics like age, gender, and the location of the lesion of interest and so on were obtained from the patients' electronic medical records after data desensitization and standardization.

2.Image preprocessing and annotation: radiographs were kept and downloaded as Digital Imaging and Communications in Medicine (DICOM) files from the picture archiving and communication system (PACS) at their original sizes and resolutions. All of these radiograph images have undergone desensitization processing of disengaging patient protected health information from DICOM data to meet the relevant legal criteria and requirements of US (HIPAA) as well as European (GDPR). Delineating the region of interest (ROI) was performed by two proficient radiologists. ROIs were meticulously outlined via Click 2 Crop (version 5.2.2) (https://click-2-crop.en.softonic.com/) to closely segment pertinent entities present in each PBT or bone infection. The smallest rectangular box that can completely cover the ROI was manually annotated as the boundary box by senior seniority radiologist to ensure accuracy. Afterwards, the annotated ROIs were used as ground truth for the model development process.

3.Design of the imaging models: for the classification of the radiographs, imaging models were built upon four distinct neural networks: EfficientNet B3 (E3), EfficientNet B4 (E4), Vision Transformer (ViT) and Swin Transformers (SWIN). Addressing the constraints of our limited label data, we adopted a transfer learning strategy. All four models were initialized with weights pre-trained on the extensive ImageNet dataset, followed by fine-tuning on our proprietary bone dataset. The original 1000-class classification head was supplanted by a singular node endowed with a sigmoid activation function, facilitating binary predictions.

4.Model training and evaluation: the internal dataset from Hospital 1 was partitioned into training, validation, and test set at a ratio of 7:1:2, respectively. The dataset from Hospital 2 and Hospital 3 was set aside as an external test set, facilitating the assessment of our models’ generalizability to data from varied sources. Each of the four models was trained independently using a batch size of 128 over 100 epochs. We employed Binary Cross-Entropy loss as our loss function. Optimization of the model was achieved through Stochastic Gradient Descent with an initial learning rate of 0.1. This rate was decayed by a factor of 10 every 30 epochs. For testing, we utilized the weights from the epoch exhibiting the best performance on the validation dataset.
Our algorithms were developed in Python 3.7 and executed on a machine equipped with an Nvidia RTX 3090 graphics processing unit. In terms of data preprocessing, all images underwent resizing and normalization. Specifically, images were resized to a resolution of 224x224 pixels and normalized using the mean and standard deviation of the training dataset. To further enhance performance, we incorporated standard data augmentation techniques during training, including random horizontal and vertical flips with a probability of 0.5 for each.

5.


# Performance

# Citation
Please cite the following paper for using: Large-Population based Deep Learning Models in Classifying Primary Bone Tumors and Bone Infections based on Radiographs:a Retrospective and Multi-reader Multi-center Study. Submission 2024.
