### The Problem
#### Voice recognition system

Develop an application which facilitates:
* Gender recognition of a person based on a voice recording.
* Age group identification of a person based on a voice recording.
* Spoken language recognition of a person based on a voice recording.

### Data
#### The dataset
For this project we've used the [Mozilla Common Voice Dataset][dataset], namely, the English dataset for gender classification and age classification. And the English, French and German datasets for language classification. The gender dataset contains 230,000 samples, labeled as male, female or other. The age dataset contains 250,000 samples labeled from teens to eighties, in tens of years increments. The language dataset is self-explanatory.

Initial gender dataset:  
![initial_gender_dataset][0]

Initial age dataset:  
![initial_age_dataset][1]

Before the data preparation step we performed some dataset preparation. After equally sampling from the dataset.
* For the gender dataset we've took only adults (age > 20).
* For the age dataset we've took only those younger than 70 years.
* For the language dataset we've took samples only from the ones who are in their
    20s and 30s.

For the training/validation of the gender model we've taken 97,000 samples split into two classes: male and female.
For the training/validation of the age model 71,000 samples were taken, split into 6 classes (teens, twenties, thirties, fourties, fifties and sixties). And 15,000 for the language model, with 5,000 from each language (english, french and german).

#### Data preparation
The audio preprocessing part and feature extraction was done using librosa.
Starting with the original audio, we've loaded it using librosa with a
sample rate of 44.1k, converting the signal to mono.  

Initial audio sample:  
![initial_audio][2]
After loading the audio, we've done some preprocess to it, removing
the background noise and the silence fragments from the audio.  
This was done in two steps:
    * apply a thresholding approach for the whole audio (silence <= 18db)
    * apply a thresholding apporach for the leading and trailing
      parts of the audio signal (silence <= 10db).
Audio after first step:
![audio_after_split_threshold][3]

Audio after second step:
![audio_after_trim_threshold][4]
With orange are the harmonics of the audio, the melody part, and with green are
the percussive parts of the audio. We've decided to keep both.

#### Audio feature extraction
We've chosen to extract from each audio file 41 features (13 MFCCs, 13 delta-MFCCs,
13 delta-delta-MFCCs, 1 pitch-estimate, 1 magnitude vector).
We've chosen 13 as the number of MFCC due to its popularity in the literature.

MFCCs, delta-MFCCs and delta-delta-MFCCs:  
![mfcc_and_deltas_from_audio][5]

Pitches and magnitudes were extracted from the audio sample using librosa, then the maximum of those values were extracted, and then a smooth function was applied (Hann function) over the pitches with a window
of 10ms.

Pitches for the audio sample:
![audio_pitches][6]

Magnitudes for the audio sample:
![audio_magnitudes][7]

### Models  

We've chose a DNN as the model due to their success in exploiting large quantities of data without overtraining and their multilevel representation.
As the dataset is composed of temporal sequences of speech, to avoid stacking audio frames as the inputs and because RNNs are prone to _gradient vanishing_ and fail to learn when time deltas exceed more than 5-10 time steps, we've chosen the Long Short-Term Memory(or LSTM) architecture.

For the purpose of evaluating the resulting models a train/validation split was made on the dataset with 80% of the samples for the training of the models and 20% for the validation.
Both the training and the validation sets were normalized to have zero mean and unit standard deviation.
Each model was optimized mini-batch-wise with a mini-batch size of 128 samples.

![lstm_cell](https://cdn-images-1.medium.com/max/800/1*z4qT1SIp79JZ21x86w_4gA.jpeg)  
LSTM cell visual representation, source: Google

#### gender classification:

For the gender classification we are using a model containing 2 LSTM layers with 100 units on each layer, the first one has a dropout of 30% applied to it and the second one a dropout of 20%, and an output layer with 2 units using softmax as the activation function.

The model is trained for 10 epochs using the _categorical crossentropy_ function as the loss function and the _ADAM_ optimizer to optimize the network weights.
After 10 epochs we've obtained a loss-score of 0.19, an accuracy of 0.93 and a precision of 0.94 on the validation set.

Gender model performance chart:
![lstm_gender_41_performance][8]

#### age classification:

For the age classification we set a LSTM with 2 hidden layers of size 256. To prevent the LSTM from overfitting, we have applied a dropout of 30% for each hidden layer. We've then added a fully-connected layer of the same size using ReLU as the activation function. Finally, the output layer has a size equal to the number of age classes using softmax as the activation function.

The model is trained for 30 epochs using the _categorical crossentropy_ function as the loss function and the _ADAM_ optimizer for weight learning.

After 30 epochs we've obtained a loss-score of 1.32, an accuracy of 0.49 and a precision of 0.68 on the validation set.

Age model performance chart:  
![lstm_age_41_performance][9]

#### language classification:

For the language classification we trained a LSTM model containing 2 hidden layers of 512 units each, and set a dropout of 60% respectively 40% to avoid model overfitting.
The output layer consists of 3 units, one for each language to be identified, and uses softmax for its activation.

The model is trained for 20 epochs using the _categorical crossentropy_ function as the loss function and the _ADAM_ optimizer for weight learning.

After 20 epochs we've obtained a loss-score of 0.8, an accuracy of 0.64 and a precision of 0.71 on the validation set.

Language model performance chart:
![lstm_lang_41_performance][10]

### Project authors:
    Forgacs Amelia
    Enasoae Simona
    Dragan Alex

[dataset]: https://voice.mozilla.org/en/datasets
[0]: images/initial_gender_dataset.png
[1]: images/initial_age_dataset.png
[2]: images/initial_loaded_audio.png
[3]: images/audio_after_split_threshold.png
[4]: images/audio_after_trim_threshold.png
[5]: images/mfcc_and_deltas_for_audio.png
[6]: images/audio_pitches.png
[7]: images/audio_magnitudes.png
[8]: images/lstm_gender_41_performance.png
[9]: images/lstm_age_41_performance.png
[10]: images/lstm_lang_41_performance.png
