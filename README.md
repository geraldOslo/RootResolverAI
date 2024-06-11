# RootResolverAI
 Use AI to do scoring of endodontic  treatment outcomes
 Repository for the Python code used in the project

## Background
Anonymized clips from periapical radiographs obtained and scored in the project Outcome of endodontic treatment at the Department of Endodontics, UiO[1] are used to train different machine learning models on PAI scoring. See repository EndodonticMeasurements [2] for details on the data aquisition.


## Data
### Image files
RGB images obtained by cropping radiographs to 224 x 224 pixels centered on the apex.
### Scoring data
CSV-files with three columns: filename, PAI, weight

## Models and training
Versions of the scripts here are snapshots. Several different amount of fine-tuning of layers, learnin rates and other hyperparameters were tested. Training data set is small and highly unbalanced:


### EfficientNet(B7)[3]
Finetuning of last 3 blocks and classifier gave a validation accuracy of 0.56. 
- Loss Function: Cross-entropy with per-sample weighting
- Optimizer: Adam (learning rate 0.0001, weight decay 1e-4)
- Scheduler: Cosine annealing for learning rate adjustment
Program code for training and evaluation [here](./code/models/EfficientNet_240529_3LFT.ipynb).

### ConvNext
More informatin be added

Program code for training and evaluation [here](./code/models/ConvNeXt_240527.ipynb).

### ResNet-50
More informatin be added

Program code for training and evaluation [here](./code/models/ResNet-50_240526.ipynb).

### VGG19
Finetuning the entire pretrained model gave a validation accuracy of 0.46. 
- Loss Function: Cross-entropy with per-sample weighting
- Optimizer: Adam (learning rate 0.0001, weight decay 1e-4)
- Scheduler: Cosine annealing for learning rate adjustment
Program code for training and evaluation [here](./code/models/VGG19_240529.ipynb).

# References
1. [Outcome of endodontic treatment at the Department of Endodontics, UiO](https://www.forskpro.uio.no/prosjekter/odont/iko/endodonti/resultatanalyse-av-endodontisk-behandling/)
2. GitHub repository: [EndodonticMeasurements](https://github.com/geraldOslo/EndodonticMeasurements)
3. Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." Proceedings of the 36th International Conference on Machine Learning (ICML), 2019




