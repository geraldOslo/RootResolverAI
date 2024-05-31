# RootResolverAI
 Use AI to do scoring of endodontic  treatment outcomes
 Repository for the Python code used in the project

# Update
Mai 31 2024 - the repository will be updated with new code in the next few days

## Background
Anonymized clips from periapical radiographs obtained and scored in the project Outcome of endodontic treatment at the Department of Endodontics, UiO[1] are used to train different machine learning models on PAI scoring. See repository EndodonticMeasurements [2] for details on the data aquisition.


## Data
### Image files
RGB images obtained by cropping radiographs to 224 x 224 pixels centered on the apex.
### Scoring data
CSV-files with three columns: filename, PAI, weight

## Models and training
### EfficientNet(B7)[3]
Finetuning of last 3 blocks and classifier gave a validation accuracy of 0.56. 
- Loss Function: Cross-entropy with per-sample weighting
- Optimizer: Adam (learning rate 0.0001, weight decay 1e-4)
- Scheduler: Cosine annealing for learning rate adjustment
Program code for training and evaluation [here](./code/models/EfficientNett_240529_3LFT.ipynb).

### ConvNext
To be added

### ResNet-50
To be added

### VGG19
To be added

# References
1. [Outcome of endodontic treatment at the Department of Endodontics, UiO](https://www.forskpro.uio.no/prosjekter/odont/iko/endodonti/resultatanalyse-av-endodontisk-behandling/)
2. GitHub repository: [EndodonticMeasurements](https://github.com/geraldOslo/EndodonticMeasurements)
3. Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." Proceedings of the 36th International Conference on Machine Learning (ICML), 2019




