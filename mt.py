import numpy as np
import load_file
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
#from keras.callbacks import Callback
class TokenizerWrap(Tokenizer):
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):

        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'
        self.num_tokens = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        self.tokens_padded = pad_sequences(self.tokens, maxlen=self.max_tokens, padding=padding, truncating=truncating)
    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 
    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]for token in tokens if token != 0]
        text = " ".join(words)
        return text
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:        
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)
        return tokens
x=load_file.load_data(english=True)
y=load_file.load_data(english=False)
num_words=20000
mark_start = 'aaaa '
mark_end = ' zzzz'
#print("aa")
tokenizer_source=TokenizerWrap(texts=x,padding='pre',reverse=True,num_words=num_words)
tokenizer_dest = TokenizerWrap(texts=y,padding='post',reverse=False,num_words=num_words)
#idx=3
tokens_src = tokenizer_source.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
token_start = tokenizer_dest.word_index[mark_start.strip()]
token_end = tokenizer_dest.word_index[mark_end.strip()]
print(tokenizer_source.tokens_to_string(tokens_src[3]))
#x_padded_en=tokenizer_source.tokens_padded
#print(tokenizer_source.tokens_to_string(x_padded[3]))
#print(x[3])
#x_padded_de=tokenizer_dest.tokens_padded
decoder_input_data = tokens_dest[:, :-1]
decoder_output_data = tokens_dest[:, 1:]
encoder_input_data = tokens_src
encoder_input = Input(shape=(None, ), name='encoder_input')
encoder_embedding = Embedding(input_dim=num_words,output_dim=128,name='encoder_embedding')
state_size=512
encoder_gru1 = GRU(state_size, name='encoder_gru1',return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',return_sequences=False)
def connect_encoder():
    net = encoder_input
    net = encoder_embedding(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
    encoder_output = net
    return encoder_output
#s=time.time()
encoder_layer=connect_encoder()
decoder_initial_state = Input(shape=(state_size,),name='decoder_initial_state')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,output_dim=128,name='decoder_embedding')
decoder_gru1=GRU(state_size, name='decoder_gru1',return_sequences=True)
decoder_gru2=GRU(state_size, name='decoder_gru2',return_sequences=True)
decoder_gru3=GRU(state_size, name='decoder_gru3',return_sequences=True)
decoder_dense = Dense(num_words,activation='linear',name='decoder_output')
def connect_decoder(thought_vector):
    net=decoder_input
    net=decoder_embedding(net)
    net=decoder_gru1(net,initial_state=thought_vector)
    net=decoder_gru2(net,initial_state=thought_vector)
    net=decoder_gru3(net,initial_state=thought_vector)
    net=decoder_dense(net)
    return net
from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1
optimizer = RMSprop(lr=1e-3)
decoder_layer=connect_decoder(encoder_layer)
machine_translator=Model(inputs=[encoder_input,decoder_input],outputs=[decoder_layer])
encoder_model=Model(inputs=[encoder_input],outputs=[encoder_layer])
decoder_layer=connect_decoder(decoder_initial_state)
decoder_model=Model(inputs=[decoder_input,decoder_initial_state],outputs=[decoder_layer])
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',verbose=1,save_weights_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./21_logs/',histogram_freq=0,write_graph=False)
callbacks = [callback_early_stopping,callback_checkpoint,callback_tensorboard]
x_train={"encoder_input":encoder_input_data,"decoder_input":decoder_input_data}
y_train={"decoder_output":decoder_output_data}
validation_split = 10000 / len(encoder_input_data)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean
machine_translator.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
#d=time.time()
#print(d-s)
#print("hhjghmghmgnmnmbmbnb")
machine_translator.load_weights(path_checkpoint)
#machine_translator.fit(x=x_train,y=y_train,batch_size=128,epochs=1,validation_split=validation_split,callbacks=callbacks)
def translate(input_text, true_output_text=None):
    input_tokens = tokenizer_source.text_to_tokens(text=input_text,reverse=True,padding=True)
    initial_state = encoder_model.predict(np.array(input_tokens))
    max_tokens = tokenizer_dest.max_tokens
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }
        decoder_output = decoder_model.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer_dest.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]
    print("Input text:")
    print(input_text)
    print()
    print("Translated text:")
    print(output_text)
    print()
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()


for idx in range(34,67):
    translate(input_text=x[idx],true_output_text=y[idx])

