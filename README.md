# Covid-19 Tweet Analysis

## Abstract
Covid19, since its inception, has had a huge impact on people's life. Since different governments have announced lock-down to confine the people at home to mitigate the spread of disease, people have turned to social media to express their concerns and their feelings about the situation. An insight into the mindset of the people is an invaluable commodity at dire times such as these. Analysis of this frail yet testing phase can give helpful and important insights in to the situation which would surely lead to better and informed decisions at higher levels to curb and contain not only the deadly disease but the rising panic and frustration. 
In this project, sentiment analysis is done on Covid19 related tweets from different parts of the world which is essentially a test classification problem. Recurrent convolution neural network (RCNN) is employed which uses a recurrent structure to capture more contextual information. This also uses a max-pooling layer to determine which words have more weight while describing sentiment of a tweet.

## Dataset
We  have  used  Covid19  UCD  Challenge  dataset.   This dataset can be categorized into 5 emotion classed namely analytical, fear, confident, anger and sadness.
#### Link: https://github.com/xxz-jessica/COVID-19_UCD_Challenge

## Proposed Solution
The first step is to clean up the raw text data.  In tweets,several stop words needs to be removed e.g.  prepositions,mentions,  hashtags,  URLs,  etc.   After a cleanup,  the dataneeds  to  be  converted  in  vector  form  to  feed  to  a  DeepNeural Network.  For the word2vec conversion, skip grammodel  is  used. This  model  learns  the  vector  representa-tion from the raw data using the similarity between severalwords  based  in  their  context. The  objective of the Skip-gram model is to learn word representations that are usefulfor predicting the nearby words in a document. Formally, given a sequence of training words/sentence, the objective of the Skip-gram model is to maximize the average log probability.

<p align="center">
  <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/w2v.png">
</p>

After the conversion of words to usable representation,the next step is to feed it to a classifier. RNN and LSTM are commonly used to extract the global information fromthe data.  RCNN on the other hand, maintains the local in-formation which signifies the prominent features within thelimited  context  of  the  document. In this way, an overallresponse  can  be  pooled  at  the  end  which  can  better  helpduring the classifications. In this model, we use a recurrentarchitecture, which is a bidirectional recurrent network, tocapture the contexts. The recurrent structure can obtain allclin a forward scan of the text andcrin a backward scan of the text. After we obtain the representation of the word, we apply a linear transformation together with the tanh activation function and send the result to the next layer. Max-pooling is applied to these which extracts the dominant features which are then passed to FC layers to classify the tweets into emotion.

<p align="center">
  <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/architecture.jpeg">
</p>

## Training Setup
   
   - Batch Size: 64
   - Embedding layers size: 100
   - Number of Enbedding layers: 3
   - Dropout: 0.41
   - Learning rate: 0.005
   - Loss: Weighted cross entropy and Focal loss
   - Optimizer: SGD
   
   ### Experiments
   - Experiment 1: Sentiment140 Dataset
        Performed on LSTM and RCNN
        - With stop words
        - Without stop words
   - Experiment 2: Covid UCD Challenge
        Performed on LSTM and RCNN
        - With stop words
        - Without stop words  
   - Experiment 3: Training model on Covid-19 UCD data using Focal Loss
   - Experiment 4: Training model on Covid-19 UCD data using Weighted Cross Entropy and Focal Loss to handle imbalanced data.
        - With stop words
        - Without stop words
   - Experiment 5: Training on Best performing model.
  
   ### Results

   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/Capture.PNG"> </p>

## Analysis

   ### Date wise trend
   <b> We analyzed the predicted tweets data to get information on how people felt (anger, sadness, fear etc) in different countries during Covid-19. 
       The tweets were gathered from Fabruary to July. Below are the figures that shows the trend. </b>
       
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Pakistan.png"> </p>    
       
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Canada.png"> </p>
   
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_India.png"> </p>
   
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_Nigeria.png"> </p>
   
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_United Kingdom.png"> </p>
   
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/datewise_country_emotion_United States.png"> </p>
   
   ### Emotion in different countries
   <b> Below is the plot that explain the emotion of people in different countries towards Covid-19. </b>
   
   <p align="center"> <img src="https://github.com/UsamaI000/G2H_Project_DLSpring2020/blob/master/images/country_emotion.png"> </p>
   
   
