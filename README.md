# ndsc_beginner

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
      <td>0.018</td>
    </tr>
    <tr>
      <td>Fashion</td>
      <td></td>
    </tr>
    <tr>
      <td>Beauty</td>
      <td></td>
    </tr>
  </tbody>
</table>

## Text Classifier


## Submission

### Total: 20
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
      <td>0.75738</td>
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
