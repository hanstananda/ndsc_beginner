# To-do List:

Here are the list of things to be done

## Data Scientist Role

- [ ]   Check maximum words in title of train data
- [ ]   Check maximum characters in title of train data
- [ ]   Check maximum words in title of test data
- [ ]   Check maximum characters in title of test data
- [ ]   Check from analysis in [kaggle](https://www.kaggle.com/chewzy/eda-for-ndsc-2019)
- [ ]   Create list of irrelevant words to be omitted in all data 
        
        List of words: ["wa", "murah", "termurah", "original", "satuan", "ready" ,"stok", "stock", "limited", "edition", "promo", "bulan", "ini", "hari", "minggu", "gratis", "ongkir", "garansi"]
        
### List of rules for the data 
- [x]   Create all characters lowercase
- [x]   Remove special characters
- [ ]   Combine some keywords for more compact search (e.g "Lip stick" into "Lipstick", "30 ml" into "30ml", "Crop top" into "croptop", "l oreal" into "loreal", "1 tahun" into "1tahun")

        Combined words and numbers already, keywords still in progress
- [ ]   Apply words omission above
- [ ]   If possible, make all words into base words based on dictionary (e.g "merajut" -> "rajut", )

## Machine Learning Designer Role

- [x]   Test Convolutional Neural Network (CNN) on words_to_sequences
        
        Result is not so promising from test result
- [x]   Test Recurrent Neural Network (RNN) on words_to_sequences

        Result is not so promising from test result
- [x]   Test CNN on words_to_sequences with divided categories

        Result is worse than regular NN with words_to_matrix
- [ ]   Test RNN on words_to_sequences with divided categories(most likely will not perfrom so good)
- [ ]   Test RNN on words_to_matrix
- [ ]   Test CNN on words_to_matrix
- [ ]   Test Hypertune parameters (Examples on `keras_text_classifier/classivier_v3.py`)
- [ ]   Try Image Classification based on 3 categories(May need to be specified further)
- [ ]   Try other words embeddings other than default Keras words_to_sequences
