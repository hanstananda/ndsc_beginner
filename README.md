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
  <tbody>
    <tr>
      <td rowspan='3'>
        <ol>
          <li>Conv2D(Conv2D(32, kernel_size=(3, 3), activation='relu')</li>
          <li>MaxPooling2D((2,2))</li>
          <li>Dropout</li>
          <li>Flatten</li>
          <li>Dense(activation='relu')</li>
          <li>Dropout</li>
          <li>Dense(activation='softmax')</li>
        </ol>
      </td>
      <td rowspan='3'>
        Remove feature extractor, use Flatten
      </td>
      <td>
        Mobile
      </td>
      <td>0.18408</td>
    </tr>
    <tr>
      <td>Fashion</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Beauty</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


## Text Classifier


## Submission

Submission|Description|Score