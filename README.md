# LANGUAGE TRANSLATOR

Using encoder decoder RNN model to tokenize input text in English language and translate it into French. Encoder model to generate a thought vector and passing the thought vector to decoder model. Thought vector providing initial state to GRU units used in the decoder which computes equivalent word tokens in another language. 


## FILE DESCRIPTION

* load_file.py is used to load dataset from computer's file system. I used europarl language dataset, link to which is http://www.statmt.org/europarl/v7/. You can also use http://www.statmt.org/europarl/.

* mt.py has line of codes both for fitting data into the model and loading checkpoints from the file sytem. To start with the language translator, comment out the line of code where we load the weights `machine_translator.load_weights(path_checkpoint)`, insted uncomment the line where we fit into our model `machine_translator.fit(x=x_train,y=y_train,batch_size=128,epochs=1,validation_split=validation_split,callbacks=callbacks)`. 

## REQUIREMENTS
* Python versions that support tensorflow
* Tensorflow
* Keras
* NVIDIA GTX 960M or above/NVIDIA GTX 1050 or above
