# A comparative analisis of training schemes for voice style transfer in spanish

[@breakermoob](https://github.com/breakermoob), [@josearangos](https://github.com/josearangos), [@cdanmontoya](https://github.com/cdanmontoya) & [@jdariasl](https://github.com/jdariasl)

## Abstract

Voice conversion is a growing field of study with the arrival of Deep Neural Network architectures that allow to transfer the voice of a source speaker to a target speaker. Most of these architectures have been tested in English. In this report we compare, through perceptual measures, the performance of a model when performing voice style transfers in Spanish using three different schemas: training it with both English and Spanish audios, fine-tuning English-learned weights with Spanish audios and finally, training the model just with Spanish audios. We also explore speaker similarity measures based on euclidean distance and Kullback-Leibler divergence to try explain the variations observed in the voice conversions between different speakers. Aditionally, we describe the application that we built to expose the best model as a web service. We conclude that for a better voice style transfer in Spanish, it is necessary to train this model directly with audios in Spanish.

## Methods

### Selected architecture
We choosed to work with the architecture proposed by [1], due to its capacity  to work with audios without transcription. Also, the authors published their [original implementation](https://github.com/jjery2243542/voice_conversion), making it easier for us to test and evaluate the architecture.

### Speakers similarity
As we experimented with the architecture, we noticed that there were performance variations depending on the source and target speakers, so we tought that it could be explained with the speakers similarity.

#### Measurement based on euclidean distance:
We used the method proposed by [3], using the audio MFCC as the characteristic vector.

 #### Measurement based on Kullback-Leibler divergence:
  We trained an UBM with all the audios. The UBM was built as a GMM and then we adapted the means vector using a MAP estimation [4]. After that, we calculated the distance between each speaker's probability distribution using the Symmetric Kullback-Leibler Divergence, defined as:

![](./src/img/eq1.png)

Where D(λ1, λ2) is the Kullback-Leiber divergence, given by

![](./src/img/eq2.png)

λi represents i-th speaker model, and O(i) is the set of T samples of the i-th speaker.


### Model development and deployment

The project setup has a frontend developed in Angular and deployed using AWS Amplify. We used the frontend to show information about our project, gather users opinions and expose the voice conversion service. The polls answers are stored into a MongoDB Atlas via a Node.js backend deployed on Heroku Dyno, and the voice conversion service was developed in FastAPI and deployed on a GCP Compute Engine with a Tesla T4 GPU. This service uses the model and audios from an AWS S3 bucket.

![](./src/img/infraestructura.png)

## Experiments

### Database
To train our models we used the corpus provided by [2] and about 1200 audios recorded in Spanish by three members of our team.

### Experimental phase
We trained four models to compare them and find which approximation brings the best results when performing a voice style transfer in Spanish.

- **M-Chou:** The first model was a replica of the training shown at [1]. We tried to get familiar with the implementation with this experiment.

- **M-Chou+3:** The second model was trained using using audios in English and in Spanish as well. With this mixed dataset we tried to check if the model was able to extract features from the audios in English and take advantage of these feature to improve the Spanish 

- **M-3SS:** The third model was trained using just the audios in Spanish. We tried to check if the model was able to extract enough features from a few speakers.

- **M-TL:** For the last model we tried to apply the *transfer learning* concept, taking the previously trained weights and *fine-tuning* them with the audios in Spanish. We replaced some layers due to the number of Spanish speakers in the training set. The best performance was achieved without freezing any layer. The replaced layers were

| Component           | Layer                        |
| ---                 | ---                          |
| SpeakerClassifier   | conv9                        |
| Decoder             | emb1, emb2, emb3, emb4, emb5 |
| Generator           | emb1, emb2, emb3, emb4, emb5 |
| PatchDiscriminator  | conv_classify                |


### Model evaluation

## Results

## References

[1] [J.-c. Chou, C.-c. Yeh, H.-y. Lee, and L.-s. Lee, “Multi-target voice conversion without parallel data by adversarially learning disentangled audio representations,” arXiv preprint arXiv:1804.02812, 2018.](https://arxiv.org/abs/1804.02812)

[2] [C. Veaux, J. Yamagishi, and K. MacDonald, CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit. University of Edinburgh. The Centre for Speech Technology Research (CSTR), 2017.](https://datashare.is.ed.ac.uk/handle/10283/2651)

[3] [M. K. Singh, N. Singh, and A. K. Singh, “Speaker’s voice characteristics and similarity measurement using euclidean distances” in 2019 International Conference on Signal Processing and Communication (ICSC), pp. 317–322, 2019.](https://ieeexplore.ieee.org/document/8938366)

[4] [D. A. Reynolds, T. F. Quatieri, and R. B. Dunn, “Speaker verificationusing adapted gaussian mixture models,” Digital Signal Processing, vol. 10, no. 1, pp. 19 – 41, 2000](https://www.sciencedirect.com/science/article/abs/pii/S1051200499903615)

# Authors
||||
|---|---|---|
|León Darío Arango Amaya | @breakermoob | leon.arango@udea.edu.co |
| Jose Alberto Arango Sánchez | @josearangos | jose.arangos@udea.edu.co |
| Carlos Daniel Montoya Hurtado | @cdanmontoya | carlos.montoya@udea.edu.co | 
| [Julián D. Arias](https://sites.google.com/site/juliandariaslondono) | @jdariasl | julian.ariasl@udea.edu.co |