# DeepTIRP
Large-Population based Deep Learning Models in Classifying  Primary Bone Tumors and Bone Infections based on Radiographs
Due to the similarity in image appearance of the lesions from primary bone tumors (PBTs) and bone infections, clinical orthopedic surgeons even professional radiologists may have difficulty distinguishing them. This study aimed to develop an ensemble deep learning algorithm differentiating PBTs from bone infections based on radiographs from multi-center. In this retrospective study, 1992 patients from three hospitals with histologically confirmed PBTs or bone infections were identified and included. Patients were divided into a training set (N=1044), a test set (N=354), an internal validation set (N=171) and an external test set (N=423). The ensemble framework was constructed using radiograph images and clinical characteristics for binary classification on PBTs and bone infections. ROC curve and confusion matrix were utilized to evaluate the classification performance, and further compared it with radiologist interpretation with different seniority. Visual analysis was conducted to explain the attention of the models via GradCAM and ScoreCAM. In the analysis conducted, the ensemble model demonstrated significant performance improvements compared to models relying solely on radiograph images. Specifically, on the internal test set, the ensemble model achieved AUCs of 0.948 (95% CI, 0.931-0.963) and accuracies of 0.881 for binary classification tasks. The consistency of the model on the external test set was demonstrated, with AUCs of 0.963 (95% CI, 0.951-0.973) and accuracies of 0.895. Meanwhile, the performance of the ensemble model was also superior to that of junior radiologist group and medium seniority group (accuracy: 75.8% and 80.2%, respectively), and comparable to that of senior radiologist group (accuracy: 83.6%). The inter-reader reliability (Fleiss κ: 0.800) between models was much higher than that between radiologists (Fleiss κ: 0.401). Our study built a framework which displayed promising performance and reliable results in classifying PBTs and bone infections. The ensemble model was better than three groups of radiologists with diverse seniority.

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
# Data collection and processing

# Performance

# Citation
Please cite the following paper for using: Large-Population based Deep Learning Models in Classifying Primary Bone Tumors and Bone Infections based on Radiographs:a Retrospective and Multi-reader Multi-center Study. Submission 2024.
