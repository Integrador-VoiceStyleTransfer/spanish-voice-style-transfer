# A comparative analisis of training schemes for voice style transfer in spanish



## Abstract

Voice conversion is a growing field of study with the arrival of Deep Neural Network architectures that allow to transfer the voice of a source speaker to a target speaker. Most of these architectures have been tested in English. In this report we compare, through perceptual measures, the performance of a model when performing voice style transfers in Spanish using three different schemas: training it with both English and Spanish audios, fine-tuning English-learned weights with Spanish audios and finally, training the model just with Spanish audios. We also explore speaker similarity measures based on euclidean distance and Kullback-Leibler divergence to try explain the variations observed in the voice conversions between different speakers. Aditionally, we describe the application that we built to expose the best model as a web service. We conclude that for a better voice style transfer in Spanish, it is necessary to train this model directly with audios in Spanish.

## Methods

### Selected architecture
We choosed to work with the architecture proposed by [1], due to its capacity  to work with audios without transcription. Also, the authors published their [original implementation](https://github.com/jjery2243542/voice_conversion), making it easier for us to test and evaluate the architecture.

### Speakers similarity

## Experiments

### Database
To train our models we used the corpus provided by [2] and about 1200 audios recorded in Spanish by three members of our team.

### Experimental phase
We trained four models to compare them and find which approximation brings the best results when performing a voice style transfer in Spanish.

- **M-Chou:** The first model was a replica of the training shown at [1]. We tried to get familiar with the implementation with this experiment.

- **M-Chou+3:**

- **M-3SS:**

- **M-TL:**

### Model evaluation

## Results

## References

[1]: [J.-c. Chou, C.-c. Yeh, H.-y. Lee, and L.-s. Lee, “Multi-target voice
conversion without parallel data by adversarially learning disentangled
audio representations,” arXiv preprint arXiv:1804.02812, 2018.](https://arxiv.org/abs/1804.02812)

[2]: [C. Veaux, J. Yamagishi, and K. MacDonald, CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit. University
of Edinburgh. The Centre for Speech Technology Research (CSTR),
2017.](https://datashare.is.ed.ac.uk/handle/10283/2651)

# Authors
||||
|---|---|---|
|León Darío Arango Amaya | @breakermoob | leon.arango@udea.edu.co |
| Jose Alberto Arango Sánchez | @josearangos | jose.arangos@udea.edu.co |
| Carlos Daniel Montoya Hurtado | @cdanmontoya | carlos.montoya@udea.edu.co | 
| [Julián D. Arias](https://sites.google.com/site/juliandariaslondono) | @jdariasl | julian.ariasl@udea.edu.co |