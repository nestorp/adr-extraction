import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import tensorflow as tf
from gensim.models import KeyedVectors
import pickle
from tb_trainval_callback import TrainValTensorBoard
from custom_metrics import Metrics
from custom_metrics import Metrics_Approx
from config import CONFIG
from ast import literal_eval
from nltk.corpus import stopwords
import string



argparser = ArgumentParser()

argparser.add_argument("--datafile", dest="datafile", default=CONFIG['datafile'],
                         help="Location of file with data and labels")
argparser.add_argument("--sentence_len", dest="sentence_len", default=CONFIG['sentence_len'],
                         help="Max length for sentences")
argparser.add_argument("--top_words", dest="top_words", default=CONFIG['top_words'],
                         help="Top N words to consider for embeddings")	
argparser.add_argument("--word2vecpath", dest="word2vecpath", default=CONFIG['word2vecpath'],
                         help="Location of binary word embedding file")	
argparser.add_argument("--num_hidden", dest="num_hidden", default=CONFIG['num_hidden'],
                         help="Number of hidden layers to use")		
argparser.add_argument("--hidden_dim", dest="hidden_dim", default=CONFIG['hidden_dim'],
                         help="Dimensions of hidden LSTM layers")	
argparser.add_argument("--lstm_act", dest="lstm_act", default=CONFIG['lstm_act'],
                         help="Activation function for LSTM layers")	
argparser.add_argument("--dense_act", dest="dense_act", default=CONFIG['dense_act'],
                         help="Activation function for dense layer")
argparser.add_argument("--optimizer", dest="optimizer", default=CONFIG['optimizer'],
                         help="Optimizer used for training")	
argparser.add_argument("--learning_rate", dest="learning_rate", default=CONFIG['learning_rate'],
                         help="Starting learning rate for training")	
argparser.add_argument("--dropout_rate", dest="dropout_rate", default=CONFIG['dropout_rate'],
                         help="Dropout rate for LSTM layers")
argparser.add_argument("--class_weights", dest="class_weights", default=CONFIG['class_weights'],
                         help="Class weights for use in training")	
argparser.add_argument("--train_embed", dest="train_embed", default=CONFIG['train_embed'],
                         help="Allows model to train embedding layer")	
argparser.add_argument("--rand_embed", dest="rand_embed", default=CONFIG['rand_embed'],
                         help="Sets the model to use randomly initialized embedding layer")		
argparser.add_argument("--max_train", dest="max_train", default=CONFIG['max_train'],
                         help="Set maximum train samples, used for learning curves")	
argparser.add_argument("--seed", dest="seed", default=CONFIG['seed'],
                         help="Random seed")
argparser.add_argument("--num_epochs", dest="num_epochs", default=CONFIG['num_epochs'],
                         help="Max number of training epochs")	
argparser.add_argument("--batch_size", dest="batch_size", default=CONFIG['batch_size'],
                         help="Training batch size")	
argparser.add_argument("--log_name", dest="log_name", default=CONFIG['log_name'],
                         help="Directory name for tensorboard logs")	
argparser.add_argument("--load_weights", dest="load_weights", default=CONFIG['load_weights'],
                         help="Determines wether to load pre-saved weights")                        

args = argparser.parse_args()

rand_embed = int(args.rand_embed)!=0
print("rand_embed", rand_embed)

train_embed = int(args.train_embed)!=0
print("train_embed", train_embed)

load_weights = int(args.load_weights)!=0
print("load_weights", load_weights)

data = pd.read_csv(args.datafile, sep="|", encoding="utf-8")

norm_text_lit = []
labels_lit = []

stop_words = set(stopwords.words('english'))

"""
for x in range(len(data)):
    try:
        wlist = data.words[x].split("~")
        llist = [z.replace("B","I") for z in data["iob"][x].split("~")]
        delindex = []
        for y in range(len(wlist)) or wlist[y] in string.punctuation:
                #print(wlist[y])
                delindex.append(y)
        del wlist[y]
        del llist[y]
        norm_text_lit.append(wlist)
        labels_lit.append(llist)
    except Exception as e:
        print(x, len(wlist), len(llist), e)
        continue
"""

for x in range(len(data)):
    try:
        #iobs = [z.replace("B","I").replace("M","O").replace("I-IND","O") for z in data["iob"][x].split("~")]
        iobs = [z.replace("B","I").replace("M","O") for z in data["iob"][x].split("~")]
        if "" in iobs:
            print(x,iobs)
        norm_text_lit.append(data.words[x].split("~"))
        labels_lit.append(iobs)
    except Exception as e:
        print(x, e)
        continue
        
print("Sentences: ", len(norm_text_lit))
all_words = list(set(np.concatenate(norm_text_lit)))

longest = 0
for sen in norm_text_lit:
    if len(sen)>longest:
        longest = len(sen)

print("Longest sentence: ", longest) 


print("Labels: ", len(labels_lit))
all_labels = list(set(np.concatenate(labels_lit)))
all_labels_full = list(np.concatenate(labels_lit))

labels_df = pd.DataFrame(all_labels_full)
print(labels_df[labels_df[0]=="O"].count())
print(labels_df[labels_df[0]=="I-ADR"].count())
print(labels_df[labels_df[0]=="I-IND"].count())
    

word_index = {w: i+1 for i, w in enumerate(all_words)}
word_index["~pad~"] = 0
all_words.insert(0,"~pad~")

all_labels.append("<PAD>")
tag_index = {t: i for i, t in enumerate(all_labels)}
pickle.dump(all_labels, open("tag_index", 'wb'))
#tag_index["<PAD>"]=2
#tag_index = {"O":0, "I-ADR": 1, "<PAD>":2}
#tag_index = {'O': 0, 'M': 1, 'I-ADR': 2,  'I-IND': 3, '<PAD>': 4}

print(tag_index)

top_words = args.top_words
num_words = len(word_index)
num_classes = len(tag_index)

print('Found %s unique tokens.' % len(word_index))

X = [[word_index[w] for w in s] for s in norm_text_lit]
y = [[tag_index[w] for w in s] for s in labels_lit]

# truncate and pad input sequences
max_review_length = longest 
X = sequence.pad_sequences(X, maxlen=max_review_length, padding='post', value=0)
y = sequence.pad_sequences(y, maxlen=max_review_length, padding='post', value=tag_index["<PAD>"])

#BINARY
#y = y.reshape(429,280,1)

#MULTICLASS
y = [to_categorical(i, num_classes=num_classes) for i in y]

print('Shape of data tensor:', X.shape)

#embedding_matrix = None

#try:
#	embedding_matrix = pickle.load(open("w2v_embedding_matrix_goog_3", 'rb'))
	#word_vectors_dim = len(embedding_matrix[0])
	#print("Found embedding matrix pickle file")
	#print("Vector length:", word_vectors_dim)
#except:
#	print("No embedding matrix found...")
	
#if embedding_matrix is None:
if not rand_embed:
    print("Loading w2v embeddings...")
    word2vecpath = args.word2vecpath
    word_vectors = KeyedVectors.load_word2vec_format(word2vecpath, binary=True, unicode_errors='replace')

    print("Finished loading w2v embeddings...")

    word_vectors_dim = len(np.asarray(word_vectors.get_vector(list(word_vectors.vocab.keys())[0]),dtype='float32'))
    print('Vector length:',word_vectors_dim)

    embeddings_index = {}
    for word in word_vectors.vocab.keys():
        coefs = np.asarray(word_vectors.get_vector(word),dtype='float32')
        embeddings_index[word] = coefs
    word_vectors = None
    print("Embedding index created...")

    embedding_matrix = np.zeros((len(word_index) , word_vectors_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
                
        #pickle.dump(embedding_matrix, open("w2v_embedding_matrix_goog_3", 'wb'))
	
    embedding_matrix=[embedding_matrix]
    print("Embedding matrix created...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)

if args.max_train is not None:
	max_train = int(args.max_train)
	if max_train<len(X_train):
		X_train = X_train[:max_train]
		y_train = y_train[:max_train]
	
print("Size of training set = ",len(X_train))

if args.optimizer.lower()=="rmsprop":
	opt = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=None, decay=0.0)
else:
	opt = "adam"
	
print("Creating Model...")
input = Input(shape=(max_review_length,))

if rand_embed:
    print("Random embeddings")
    embedding_matrix = None
    word_vectors_dim = 400

#Add the embedding layer	
model = Embedding(input_dim = num_words, output_dim = word_vectors_dim, input_length=max_review_length, weights = embedding_matrix, trainable=train_embed, mask_zero=True)(input)

#model = Masking(mask_value=0.0)(model)

dropout_rate = int(args.dropout_rate)
num_hidden = int(args.num_hidden)

#Add Additional hidden LSTM layers
#if num_hidden>1:
#	for x in range(1, num_hidden):
		#model = Bidirectional(LSTM(int(args.hidden_dim), 
                    #batch_input_shape=(32, None, 280), 
         #           return_sequences = True, 
          #          dropout =dropout_rate, activation=args.lstm_act))(model)

if load_weights:    
    print("Loading saved weights")
    bidirectional_1_weights = np.load("bidirectional_1_weights.npy")
else:
    bidirectional_1_weights = None
          
#Add last hidden LSTM layer
model = Bidirectional(LSTM(int(args.hidden_dim), 
                    weights = bidirectional_1_weights,
                    #batch_input_shape=(32, None, 280), 
                    return_sequences = True, 
                    dropout =dropout_rate, activation=args.lstm_act))(model)

"""                   
#Add Additional hidden LSTM layers
if num_hidden>1:
	for x in range(1, num_hidden):
		model = Bidirectional(CuDNNLSTM(int(args.hidden_dim), 
                    #batch_input_shape=(32, None, 280), 
                    return_sequences = True))(model)

#Add last hidden LSTM layer
model = Bidirectional(CuDNNLSTM(int(args.hidden_dim), 
                    #batch_input_shape=(32, None, 280), 
                    return_sequences = True))(model)                
"""
#Add final dense layer BINARY
#out = TimeDistributed(Dense(1, activation=args.dense_act, ))(model)

#Add final dense layer MULTICLASS
out = TimeDistributed(Dense(num_classes, activation=args.dense_act, ))(model)

model = Model(input,out)

#BINARY
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

#MULTICLASS
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

print(model.summary())

metrics = Metrics_Approx()

if len(X_train)<int(args.batch_size):
	batch_size = len(X_train)
else:
	batch_size = int(args.batch_size)

class_weight = args.class_weights

history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=int(args.num_epochs), verbose=1, #class_weight={0:1, 1:10},
					callbacks = [metrics],
					validation_data=(X_test, np.array(y_test)))


# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#all_words.append("None")
if True:
    for i in range(10):
        #i = 1
        p = model.predict(np.array([X_test[i]]))
        p = np.argmax(p, axis=-1)
        print("{:25} ({:6}): {}".format("Word", "True", "Pred"))
        for w, l, pred in zip(X_test[i], y_test[i], p[0]):
            if w>0:
                l = np.argmax(l, axis=-1)
                print("{:25} {:6}:{}".format(all_words[w], all_labels[l], all_labels[pred]))
        print("---------------------------------------")
        
