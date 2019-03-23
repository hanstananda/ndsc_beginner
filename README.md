# NDSC Beginner Experimental Results and Submissions

## Image Classifier

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Category</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody id="table-body">
    <tr>
      <td rowspan="3">
        <ol>
          <li>Conv2D(32, kernel_size=(3, 3), activation="relu")</li>
          <li>MaxPooling2D(pool_size=(2, 2))</li>
          <li>Dropout(0.2)</li>
          <li>Flatten()</li>
          <li>Dense(128, activation="relu")</li>
          <li>Dropout()</li>
          <li>Dense(activation="softmax")</li>
        </ol>
      </td>
      <td rowspan="3">Remove feature Extractor, use Flatten</td>
      <td>Mobile</td>
      <td>
        <ul>
          <li>
            0.018 (100 data per class)
          </li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Fashion</td>
      <td>
        <ul>
          <li>0.15789 (100 data per class)</li>
          <li>0.0547 (2000 data per class)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Beauty</td>
      <td><i>Not tested</i></td>
    </tr>
  </tbody>
</table>

## Text Classifier

> Scores are general for all categories

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody id="text-classifier">
    <tr>
      <td>
        <ol>
          <li>Embedding(len(word_index) + 1,300, input_length=max_length, weights=[embedding_matrix], trainable=True)
          </li>
          <li>Flatten()</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>Embedding with normal Dense NN Softmax</td>
      <td>0.71885</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Bidirectional(CuDNNLSTM(100))</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 2.0 Embedding with LSTM RNN, training time is freaking longer than others w/o CUDNNLSTM</td>
      <td>0.7244</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Bidirectional(CuDNNLSTM(128, return_sequences=True))</li>
          <li>Bidirectional(CuDNNLSTM(128))</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(Dense(num_classes, activation="softmax"))</li>
        </ol>
      </td>
      <td>model 2.1, adding two CuDNNLSTM</td>
      <td>0.73003</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Bidirectional(CuDNNGRU(128, return_sequences=True))</li>
          <li>Bidirectional(CuDNNGRU(128))</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 2.2, after 15 epochs, converges after that</td>
      <td>0.72782</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Conv1D(128, 5, activation="relu")</li>
          <li>GlobalMaxPooling1D()</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 3: Embedding with Convolutional NN, seems a bit less likely to increase</td>
      <td>0.71</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Conv1D(128, 5, activation="relu")</li>
          <li>MaxPooling1D(5)</li>
          <li>Conv1D(128, 5, activation="relu")</li>
          <li>MaxPooling1D(5)</li>
          <li>Conv1D(128, 5, activation="relu")</li>
          <li>MaxPooling1D(5)</li>
          <li>Flatten()</li>
          <li>Dense(512, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 3.1: Embedding with multilevel CNN, very long time to train</td>
      <td>0.71</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Embedding()</li>
          <li>SpatialDropout1D(0.2)</li>
          <li>Bidirectional(CuDNNLSTM(128, return_sequences=True))</li>
          <li>Bidirectional(CuDNNLSTM(128, return_sequences=True))</li>
          <li>Conv1D(256, 5, activation="relu")</li>
          <li>GlobalMaxPooling1D()</li>
          <li>Dense(256, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 4.0</td>
      <td>0.73909</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Embedding()</li>
          <li>Dropout(0.25)</li>
          <li>Conv1D(256, 5, activation="relu", padding="valid", strides=1)</li>
          <li>MaxPooling1D(pool_size=4)</li>
          <li>CuDNNLSTM(256)</li>
          <li>Dense(256, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 4.1: may stil slightly increase</td>
      <td>0.734</td>
    </tr>
    <tr>
      <td>
        <ol>
          <li>Embedding()</li>
          <li>Dropout(0.25)</li>
          <li>TimeDistributed(Conv1D(256, 5, activation="relu", padding="same", strides=1))</li>
          <li>TimeDistributed(MaxPooling1D(pool_size=4))</li>
          <li>TimeDistributed(Conv1D(256, 5, activation="relu", padding="same", strides=1))</li>
          <li>TimeDistributed(MaxPooling1D(pool_size=2))</li>
          <li>SpatialDropout1D(0.2)</li>
          <li>Bidirectional(CuDNNLSTM(128, return_sequences=True))</li>
          <li>Bidirectional(CuDNNLSTM(128, return_sequences=False))</li>
          <li>Dense(256, activation="relu")</li>
          <li>Dropout(0.5)</li>
          <li>Dense(num_classes, activation="softmax")</li>
        </ol>
      </td>
      <td>model 4.2: a bit worse than model 4.0</td>
      <td>0.724</td>
    </tr>
  </tbody>
</table>


## Submission

### Total: 21
<table>
  <thead>
    <tr>
      <th>Submission</th>
      <th>Note</th>
      <th>Score</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody id="submission">
    <tr>
      <td>res_aggg.csv</td>
      <td></td>
      <td>0.76076 :white_check_mark:</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_5000_03_23_2019_20_40_08.csv</td>
      <td>Final fine-tuning test from old-classifier.py</td>
      <td>0.75918</td>
      <td></td>
    </tr>
    <tr>
      <td>res_aggg.csv</td>
      <td>aggregate of results</td>
      <td>0.01187</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_8_25000_03_22_2019_23_12_47.csv</td>
      <td>change max word to 25000</td>
      <td>0.75651</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_7500_03_21_2019_23_44_38.csv</td>
      <td>add num words</td>
      <td>0.75761</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_10_5000_03_21_2019_22_40_39.csv</td>
      <td>Test Add 1 layer Dense</td>
      <td>0.75850</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_6000_03_21_2019_02_40_55.csv</td>
      <td></td>
      <td>0.75912</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_5000_03_20_2019_14_01_17.csv</td>
      <td>Test max words to 5000</td>
      <td>0.75933 :white_check_mark:</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_5000_03_20_2019_14_01_17.csv</td>
      <td>Test max words to 5000</td>
      <td>0.75933</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_2500_03_19_2019_23_13_07.csv</td>
      <td>Change Glove to FastText Embedding</td>
      <td>0.75421</td>
      <td></td>
    </tr>
    <tr>
      <td>submission.csv</td>
      <td>Add 1 flatten layer, 512D</td>
      <td>0.75873</td>
      <td></td>
    </tr>
    <tr>
      <td>submission.csv</td>
      <td>test</td>
      <td>0.73973</td>
      <td></td>
    </tr>
    <tr>
      <td>res_epoch_9_03_19_2019_01_42_28.csv</td>
      <td>bidirectional 1024, but input is test.csv instead of new_test.csv</td>
      <td>0.74124</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_3500_03_17_2019_18_01_56.csv</td>
      <td>LSTM-&gt;2CNN Glove Cleaned Epoch 9 512D layers</td>
      <td>0.75738 :white_check_mark:</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_3500_03_17_2019_03_20_14.csv</td>
      <td>LSTM-&gt;CNN Glove Cleaned Epoch 9 512D layers</td>
      <td>0.75446</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_2500_03_17_2019_02_53_25.csv</td>
      <td>Flip LSTM and CNN Position</td>
      <td>0.74947</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_9_2500_03_17_2019_01_54_11.csv</td>
      <td>LSTM+CNN+Cleaned+Glove</td>
      <td>0.7573</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_8_2500_03_14_2019_23_18_20.csv</td>
      <td>Test_reload_from_h5</td>
      <td>0.70883</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_8_2500_03_14_2019_22_56_14.csv</td>
      <td>New clean</td>
      <td>0.75442</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_12_5000_03_13_2019_21_46_10.csv</td>
      <td>Cleaned data version 2 LSTM Glove</td>
      <td>0.74332</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_10_5000_03_13_2019_22_03_14.csv</td>
      <td>Cleaned data version 2 LSTM Glove top 5000 words</td>
      <td>0.71055</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_10_2500_03_13_2019_02_59_15.csv</td>
      <td>data cleaning fix1</td>
      <td>0.75413</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_12_2500_03_12_2019_18_50_52.csv</td>
      <td>Text classifier updated</td>
      <td>0.63379</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_10_2500_03_04_2019_01_55_08.csv</td>
      <td>data cleaned</td>
      <td>0.73182</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_10_2500_03_03_2019_02_01_49.csv</td>
      <td>ML CNN 10 epoch with specialization</td>
      <td>0.74267</td>
      <td></td>
    </tr>
    <tr>
      <td>resepoch_30_2500_02_28_2019_00_46_38.csv</td>
      <td>Domain-specific training ML text classification</td>
      <td>0.75224</td>
      <td></td>
    </tr>
    <tr>
      <td>res7.csv</td>
      <td>ML text classification trained with more epoch</td>
      <td>0.73401</td>
      <td></td>
    </tr>
    <tr>
      <td>res5.csv</td>
      <td>ML text classifier</td>
      <td>0.73343</td>
      <td></td>
    </tr>
    <tr>
      <td>res.csv</td>
      <td>Text classifier updated</td>
      <td>0.63379</td>
      <td></td>
    </tr>
    <tr>
      <td>out.csv</td>
      <td>text_analysis_1</td>
      <td>0.6255</td>
      <td></td>
    </tr>
  </tbody>
</table>
