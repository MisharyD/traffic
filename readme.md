I first test it with only one hidden layer and 1 conv and 1 pooling layers which has 128 units aprrox 0.2 acc.
then adden another layer with 256 units and got 0.65 acc.

I put the first layer of the 2 to 256 units and got less accuracy, I dont know how this works yet.

adding another hidden layer with 512 units and and a drop out didnt make any difference.
noticed that having 3 hidden layers worsened the accuracy, with 2 I get 0.6 and with 3 I get 0.15.

even with more setting it to train 100 times did not improve. Note: this is all with the small dataset.

removing the pooling layer noticable increased the training time, but increased the accuracy,

returning back to dense layer with 128 units and 1 conv and pooling layers.
lowering the dropout to 0.3 improved the acc from 5% to 60 %.
removing the drop out decressed the acc to 20%. what is happening.

trying with the big dataset all are 10 epochs:
1 pooling 1 conv 1 dense layer with 128 units with 0.5 droput: 5%acc.. damn

1 pooling 1 conv 1 dense layer with 128 units with 0.3 droput: improvment to 48%

1 pooling 1 conv 1 dense layer with 128 units with 0.1 droput: still 48% what is happening

1 pooling 1 conv 1 dense layer with 128 units and another 112 units layer with 0.3 droput: still 48% what is happening

1 pooling 1 conv 1 dense layer with 128 units and another 112 units layer with 0.3 droput: 4%acc... it seems that I still dont understand it fully, I will return to this later
