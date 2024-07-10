# Traffic
Traffic is neural network bult using tensor flow that can classify road signs based on images of these road signs. It is trained using the GTSRB data set. <br>
The model generated in the program can classify road signs with 99% accuracy. The process to get to this accuracy is provided below:

#### All of the below is using the gtsrb dataset, relu activiation for the hidden layer, soft max for the output layer and the following compiler: model.compiler(optimizer='adam'loss='categorical_crossentropy',metrics=['accuracy'])

10 epoch:<br>
    0.5 dropout:<br>
        1 conv with 32 filters and 2,2 pooling layers:<br>
            1 dense hidden layer with 128 units: after a couple of runs, it ranges from 85 to 95% and one time I got 45%.<br> 

20 epoch:<br>
    0.5 dropout:<br>
        1 conv with 32 filters and 2,2 pooling layers:<br>
            1 dense hidden layer with 128 units: ranges from 93 to 96%, more consistent.<br>

cross validation:<br>
    20 epoch:<br>
        0.5 dropout:<br>
            1 conv with 32 filters and 2,2 pooling layers:<br>
                1 dense hidden layer with 128 units: got 87% average across all folds 2 times and 97% 2 times. maybe the starting weights of the 87% were not so lucky ?<br>

#### after asking gpt why it were not stable he suggestedd Normalizing pixel values for images so I did. also added : np.random.seed(1) tf.random.set_seed(2)

Normalize pixel values to [0, 1]:<br>
    cross validation:<br>
        20 epoch:<br>
            0.5 dropout:<br>
                1 conv with 32 filters and 2,2 pooling layers:<br>
                    1 dense hidden layer with 128 units: siginificant improvmenet with a stable 98% accuracy.<br> 

#### reason for improvemnet is explained here: https://stackoverflow.com/questions/62111708/what-is-the-significance-of-normalization-of-data-before-feeding-it-to-a-ml-dl-m

#### From now on, the normilizing of pixel values and 20 epochs is included on all tests 

#### returning to test without cross validiation for time saving purposes.
seeds set:<br>
        0.5 dropout:<br>
            1 conv with 32 filters and 2,2 pooling layers:<br>
                1 dense hidden layer with 128 units: stable 97% to 98% accuracy.<br>

0.5 dropout:<br>
    1 conv with 32 filters and 2,2 pooling layers:<br>
        1 dense hidden layer with 128 units: stable 97% to 98% accuracy.<br>

#### Now I will try playing with the layers to reach the 99% accuracy.

0.5 dropout:<br>
    1 conv with 32 filters and 2,2 pooling layers:<br>
        1 dense hidden layer with 256 units: no difference really, still stable 97% to 98% accuracy.<br>

cross validation:<br>
        20 epoch:<br>
            0.5 dropout:<br>
                1 conv with 32 filters and 2,2 pooling layers:<br>
                    1 dense hidden layer with 256 units: average of 98% across 5 folds<br>

#### 256 untis seems not to make any difference

cross validation:<br>
        20 epoch:<br>
            0.5 dropout:<br>
                1 conv with 32 filters and 2,2 pooling layers:<br>
                    1 dense hidden layer with 128 units<br>
                    and another dense hidden layer with 128 units: 97 to 98 average across 5 folds<br>

#### it seems modifying the hidden layers has no much difference, so I will return to 1 hidden layer with 256 units

cross validation:<br>
        20 epoch:<br>
            0.3 dropout:<br>
                1 conv with 32 filters and 2,2 pooling layers:<br>
                    1 dense hidden layer with 128 units:average of 98% across 5 folds<br>

cross validation:<br>
        20 epoch:<br>
            0.1 dropout:<br>
                1 conv with 32 filters and 2,2 pooling layers:<br>
                    1 dense hidden layer with 128 units:average of 98% across 5 folds<br>

#### it also seems that dropout does not make that difference, I will start playing with the conv layers

cross validation:<br>
        20 epoch:<br>
            0.1 dropout:<br>
                2 conv with 32 filters and 2 with 2,2 pooling layers:<br>
                    1 dense hidden layer with 128 units: finally average of 99% across 5 folds<br>

#### I will stop here, here is what to conclude: changing the hidden layers and dropout did not make any difference. normilzing the pixel values made a huge differnce by ropugly 2 or 3% and got rid of the sudden 50% accuracy. adding another conv layer made the jump to 99%.
                
    











