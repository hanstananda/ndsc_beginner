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

Submission|Description|Score