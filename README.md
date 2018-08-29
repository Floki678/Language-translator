# LANGUAGE TRANSLATOR
Using encoder decoder neural net architecture to translate french to english


## FILE DESCRIPTION

* load_file.py is used to load dataset from computer's file system. I used europarl language dataset, link to which is http://www.statmt.org/europarl/v7/. You can also use http://www.statmt.org/europarl/.

* mt.py has line of codes both for fitting data into the model and loading checkpoints from the file sytem. To start with the language translator, comment out the line of code where we load the weights `machine_translator.load_weights(path_checkpoint)`, insted uncomment the line where we fit into our model `machine_translator.fit(x=x_train,y=y_train,batch_size=128,epochs=1,validation_split=validation_split,callbacks=callbacks)`. 
