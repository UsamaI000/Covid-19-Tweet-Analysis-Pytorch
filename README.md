# Covid-19 Tweet Analysis

## Members
  - Muhammad Usama Irfan
  - Abdul Basit
  - Obaid
  - Muhammad Shehryar
  - Hadi Mustafa

## Introduction
Covid19, since its inception, has had a huge impact on people's life. Since different governments have announced lock-down to confine the people at home to mitigate the spread of disease, people have turned to social media to express their concerns and their feelings about the situation. An insight into the mindset of the people is a valuable commodity at dire times such as these. Analysis of this frail yet testing phase can give helpful and important insights in to the situation which would surely lead to better and informed decisions at higher levels to curb and contain not only the deadly disease but the rising panic and frustration. 
In this project, sentiment analysis is done on Covid19 related tweets from different parts of the world which is essentially a test classification problem. Recurrent convolution neural network (RCNN) is employed which uses a recurrent structure to capture more contextual information. This also uses a max-pooling layer to determine which words have more weight while describing sentiment of a tweet.

## Guide to use
  - Download the files
  - Open Data preparation and make word2vec of your data and save .npz files and other necessary files for training.
  - Open Config.py and adjust it according to your setup and adjust related file paths in it.
  - To modify training loss and optimizer open train.py and edit it.
  - To start training run train.py
  - To test your model or get predictions of unseen data run test.py and predict.py
  - To demonstrate your model in real time you can run tweet.py and give it a random Covid related tweet

## Dataset

  ### Training
  <p> We  have  used  Covid-19  UCD  Challenge  dataset to train our RCNN model. This dataset can be categorized into 5 emotion classes namely analytical, fear, confident, anger   and sadness. </p>
  <b> Link: https://github.com/xxz-jessica/COVID-19_UCD_Challenge </b>
  <br/>
  <br/>
  <p align="center"> <img width=700 height= 350 src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/word_cloud_anger.png"> </p>
  
  ### Prediction
  <p> We used the trained model to predict on the unseen tweets dataset which was about 17M from which almost 1.5M tweets had information about countries. We used this 
  data to analyze people's feelings, attitude towards Covid. Also, we analyzed deaths per day in Countries and Date-wise sentiment analysis. </p>
  <b> Link: https://drive.google.com/drive/folders/1dVr4yYlptJefiooO_lyvGzyPSha44QNF?usp=sharing </b>

## Proposed Solution
The first step is to clean up the raw text data.  In tweets,several stop words needs to be removed e.g.  prepositions,mentions,  hashtags,  URLs,  etc.   After a cleanup,  the dataneeds  to  be  converted  in  vector  form  to  feed  to  a  DeepNeural Network.  For the word2vec conversion, skip grammodel  is  used. This  model  learns  the  vector  representation from the raw data using the similarity between severalwords  based  in  their  context. The  objective of the Skip-gram model is to learn word representations that are usefulfor predicting the nearby words in a document. Formally, given a sequence of training words/sentence, the objective of the Skip-gram model is to maximize the average log probability.

<p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/w2v.png"> </p>

After the conversion of words to usable representation,the next step is to feed it to a classifier. RNN and LSTM are commonly used to extract the global information fromthe data.  RCNN on the other hand, maintains the local in-formation which signifies the prominent features within thelimited  context  of  the  document. In this way, an overall response  can  be  pooled  at  the  end  which  can  better  helpduring the classifications. In this model, we use a recurrent architecture, which is a bidirectional recurrent network, to capture the contexts. The recurrent structure can obtain all context in a forward scan of the text and context in a backward scan of the text. After we obtain the representation of the word, we pass it to the Max-pool layer which gets the most dominant features which are then passed to the FC layer to get classified.

<p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/architecture.jpeg"> </p>

## Training Setup
We used this RCNN model to train it on the Covid-19 UCD dataset which had five emotion classes i.e. anger, fear, sadness, confident and analytical. We performed a total of 5 experiments. Initial two experiments were to make comparison of LSTM and RCNN on a twitter sentiment dataset i.e. Sentiment140. Other experiments were done of Covid-19 UCD data with two focal losses which are Cross Entropy and Focal Loss. The last experiment is done using Weighted Cross Entropy to handle dataset imbalance.

Following configurations were used for final model training.
  - Batch Size: 64
  - Embedding Dimension: 300
  - Embedding Layers: 3
  - Embedding Layers size: 100
  - Learning rate: 0.005
  - Optimizer: SGD
  - Loss: Weighted Cross Entropy

  ### Experiments
   - Experiment 1: Sentiment140 Dataset Performed on LSTM and RCNN
      - With stop words
      - Without stop words
   - Experiment 2: Covid UCD Challenge Performed on LSTM and RCNN
      - With stop words
      - Without stop words
   - Experiment 3: Training model on Covid-19 UCD data using Focal Loss
   - Experiment 4: Training model on Covid-19 UCD data using Weighted Cross Entropy and Focal Loss to handle imbalanced data.
     - With stop words
     - Without stop words
   - Experiment 5: Training on Best performing model.

## Results
<p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/Capture.PNG"> </p>

## Analysis

  ### Date wise trend
  We analyzed the predicted tweets data to get information on how people felt (anger, sadness, fear etc) in different countries during Covid-19. The tweets were gathered from     Fabruary to July. Below are the figures that shows the trend.

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Pakistan.png"> </p>


  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Canada.png"> </p>


  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_India.png"> </p>


  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Nigeria.png"> </p>


  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_United Kingdom.png"> </p>


  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_United States.png"> </p>


  ### Emotion in different countries
  Below is the plot that explain the emotion of people in different countries towards Covid-19. The plot shows that the most of the people in different countries were confident   during this time of Global pandemic. There was an emotion of fear which kept changing during the timeline.

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/country_emotion.png"> </p>


  ### Deaths in countries
  We also performed analysis of deaths per day due to Corona Virus in different countries. The graphs are given to see the trend.

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/pakistan.PNG"> </p>
  <br/>
 
  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/canada.PNG"> </p>
  <br/>

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/india.PNG"> </p>
  <br/>

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/nigeria.PNG"> </p>
  <br/>

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/uk.PNG"> </p>
  <br/>

  <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/us.PNG"> </p>
