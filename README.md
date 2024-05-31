# RootResolverAI
 Use AI to do scoring of endodontic  treatment outcomes
 Repository for the Python code used in the project

# Update
Mai 31 2024 - the repository will be updated with new code in the next few days

## Background
Periapical radiographs obtained and scored in the project <to come> are used to train different machine learning models on PAI scoring. See this repository [EndodonticMeasurements][1] details on the data aquisition.


This is a reference to [Google][1].

[1]: https://www.google.com

## Data
### Image files
RGB images obtained by cropping radiographs to 224 x 224 pixels centered on the apex.
### Scoring data
CSV-files with three columns: filename, PAI, weight

## Models and training
### EfficientNet(B7)
Finetuning of last 3 blocks and classifier gave a validation accuracy of 0.56. 
- Loss Function: Cross-entropy with per-sample weighting
- Optimizer: Adam (learning rate 0.0001, weight decay 1e-4)
- Scheduler: Cosine annealing for learning rate adjustment

### ConvNext

### ResNet-50

### VGG19

# References
[1]: https://github.com/geraldOslo/EndodonticMeasurements



