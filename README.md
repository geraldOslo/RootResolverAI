# RootResolverAI
 Use AI to do scoring of endodontic  treatment outcomes
 
 Repository for the Python code used in the project

## Update 18.09.2023
Have been to [PRESIMAL Autum Research School](https://mmiv.no/presimal/) and got so much valuable feedback.
- moved all old code to folder old
- started new approach using Pytorch (not because it's better than Tensorflow, but seems to be used more by the community)
- did proper normalisation (not only divide by 255 but same mean and stdev for all images)
- new model is learning
- new model is still not good on test data (overfitting?)

## TODO
- Try other models RESUNET is a promising candidate
- Try to use more data at once
- Make more high quality, high trust examples to learn
