# Covid-19 Tweet Analysis

## Abstract
Covid19, since its inception, has had a huge impact on people's life. Since different governments have announced lock-down to confine the people at home to mitigate the spread of disease, people have turned to social media to express their concerns and their feelings about the situation. An insight into the mindset of the people is an invaluable commodity at dire times such as these. Analysis of this frail yet testing phase can give helpful and important insights in to the situation which would surely lead to better and informed decisions at higher levels to curb and contain not only the deadly disease but the rising panic and frustration. 
In this project, sentiment analysis is done on Covid19 related tweets from different parts of the world which is essentially a test classification problem. Recurrent convolution neural network (RCNN) is employed which uses a recurrent structure to capture more contextual information. This also uses a max-pooling layer to determine which words have more weight while describing sentiment of a tweet.

## Dataset
We  have  used  Covid19  UCD  Challenge  dataset.   Thisdataset can be categorized into 5 emotion classed namelyanaltical, fear, confident, anger and sadness.
#### Link: https://github.com/xxz-jessica/COVID-19_UCD_Challenge

## Proposed Solution
The first step is to clean up the raw text data.  In tweets,several stop words needs to be removed e.g.  prepositions,mentions,  hashtags,  URLs,  etc.   After a cleanup,  the dataneeds  to  be  converted  in  vector  form  to  feed  to  a  DeepNeural Network.  For the word2vec conversion, skip grammodel  is  used.   This  model  learns  the  vector  representa-tion from the raw data using the similarity between severalwords  based  in  their  context.   The  objective  of  the  Skip-gram model is to learn word representations that are usefulfor predicting the nearby words in a document.  Formally,given a sequence of training words/sentencew1,w2,w3,... ,wT, the objective of the Skip-gram model is to maximizethe average log probability.

After the conversion of words to usable representation,the next step is to feed it to a classifier. RNN and LSTM are commonly used to extract the global information fromthe data.  RCNN on the other hand, maintains the local in-formation which signifies the prominent features within thelimited  context  of  the  document. In this way, an overallresponse  can  be  pooled  at  the  end  which  can  better  helpduring the classifications. In this model, we use a recurrentarchitecture, which is a bidirectional recurrent network, tocapture the contexts. The recurrent structure can obtain allclin a forward scan of the text andcrin a backward scan of the text. After we obtain the representation of the word, we apply a linear transformation together with the tanh activation function and send the result to the next layer. Max-pooling is applied to these which extracts the dominant features which are then passed to FC layers to classify the tweets into emotion.

![](images/architecture.jpeg) 

## Experiment setup
   
   - Batch Size: 64
   - Embedding layer size: 100
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
   - Experiment 4: Training model on Covid-19 UCD data using Weighted Cross Entropy and Focal Loss to handle imbalance data.
   - Experiment 5: Training on Best performing model.
  
## Results

  #### Using Weighted Cross Entropy
   
   Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training F1 Score | Validation F1 Score
   ------------ | ---------------- | ----------------  | ----------------    | ----------------  | ----------------
   0.30622      | 0.35316          | 90.2%             | 88.0%               | 0.9031            | 0.8836
   
  #### Using Weighted Focal Loss
   
   Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training F1 Score | Validation F1 Score
   ------------ | ---------------- | ----------------  | ----------------    | ----------------  | ----------------
   0.0268       | 0.03331          | 87.4%             | 86.90%              | 0.8849            | 0.8796
   
  <style type="text/css">
.tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
.tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-7f04{background-color:#409cff;border-color:inherit;color:#ffffff;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-crdk{background-color:#409cff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-fymr">Experiments</th>
    <th class="tg-7btt">Model</th>
    <th class="tg-7btt">Stop Words</th>
    <th class="tg-7btt">Loss</th>
    <th class="tg-7btt">Epochs</th>
    <th class="tg-7btt">Train loss</th>
    <th class="tg-7btt">Valid loss</th>
    <th class="tg-7btt">Train Accuracy</th>
    <th class="tg-7btt">Valid Accuracy</th>
    <th class="tg-7btt">Train F1</th>
    <th class="tg-7btt">Valid F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7f04" rowspan="2"><br><br>Experiment 1<br></td>
    <td class="tg-7btt">LSTM</td>
    <td class="tg-7btt">Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">25</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.411</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.499</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">82</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">78</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7btt">RCNN</td>
    <td class="tg-7btt">Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">25</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.399</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.463</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">82.9</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">79</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7f04" rowspan="4"><br><br><br><br><br>Experiment 2</td>
    <td class="tg-7btt" rowspan="2"><br>LSTM</td>
    <td class="tg-7btt">Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">55</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.2282</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.585</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">92</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">79.8</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7btt">Not Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">55</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.2782</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.656</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">91.2</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">74.4</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7btt" rowspan="2"><br>RCNN</td>
    <td class="tg-7btt">Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">55</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.275</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.470</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">91.8</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">81.8</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7btt">Not Removed</td>
    <td class="tg-7btt">Cross Entropy</td>
    <td class="tg-c3ow">55</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.369</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.647</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">89.5</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">76.56</span><br></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-7f04">Experiment 3</td>
    <td class="tg-7btt">RCNN</td>
    <td class="tg-7btt">Removed</td>
    <td class="tg-7btt">Focal Loss</td>
    <td class="tg-c3ow">50</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.0364</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.0438</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">85.09</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">82.6</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">0.8518</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.8307</span><br></td>
  </tr>
  <tr>
    <td class="tg-7f04" rowspan="4"><br><br><br><br><br>Experiment 4</td>
    <td class="tg-7btt" rowspan="4"><br><br><br><br><br>RCNN</td>
    <td class="tg-7btt" rowspan="2"><br><br>Removed</td>
    <td class="tg-7btt">Weighted Focal Loss<br></td>
    <td class="tg-c3ow">65</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.0537</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.0637</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">85.7</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">82.1</span><br></td>
    <td class="tg-c3ow">0.859</td>
    <td class="tg-c3ow">0.821</td>
  </tr>
  <tr>
    <td class="tg-7btt">Weighted Cross Entropy</td>
    <td class="tg-c3ow">65</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.296</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.531</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">88.9</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">82.9</span><br></td>
    <td class="tg-c3ow">0.889</td>
    <td class="tg-7btt">0.829</td>
  </tr>
  <tr>
    <td class="tg-7btt" rowspan="2"><br><br>Not Removed</td>
    <td class="tg-7btt">Weighted Focal Loss</td>
    <td class="tg-c3ow">65</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.022</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.032</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">88.4</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">87</span></td>
    <td class="tg-c3ow">0.8895</td>
    <td class="tg-c3ow">0.871</td>
  </tr>
  <tr>
    <td class="tg-7btt">Weighted Cross Entropy</td>
    <td class="tg-c3ow">65</td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.268</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">0.390</span><br></td>
    <td class="tg-c3ow"><span style="color:#000;background-color:transparent">89.9</span><br></td>
    <td class="tg-7btt"><span style="color:#000;background-color:transparent">87.9</span><br></td>
    <td class="tg-c3ow">0.899</td>
    <td class="tg-7btt">0.879</td>
  </tr>
  <tr>
    <td class="tg-crdk"><span style="color:#FFF">Experiment 5</span></td>
    <td class="tg-7btt">RCNN</td>
    <td class="tg-7btt">Not Removed</td>
    <td class="tg-7btt">Weghted Cross Entropy</td>
    <td class="tg-c3ow">55</td>
    <td class="tg-c3ow">0.30622</td>
    <td class="tg-c3ow">0.3511</td>
    <td class="tg-c3ow">89.2</td>
    <td class="tg-7btt">88.13</td>
    <td class="tg-c3ow">0.903</td>
    <td class="tg-7btt">0.883</td>
  </tr>
</tbody>
</table>
