######
# Drug_name_pred.py
# 
# Author: Nestor Prieto Chavana
# Date: 10/8/2018
#
# This script performs is used for the transfer learning task of the project
# The model in this script is configured to predict the name of a drug from
# the context in a post. Weights are saved and then loaded in the script
# sequence_tagging.py to perform transfer learning
#

###
import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from gensim.models import KeyedVectors
from custom_metrics import Metrics
from config import CONFIG
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

#Parsing command line arguments
argparser = ArgumentParser()

argparser.add_argument("--datafile", dest="datafile", default=CONFIG['datafile'],
                         help="Location of file with data and labels")
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
argparser.add_argument("--show_samples", dest="show_samples", default=CONFIG['show_samples'],
                         help="Determines wether to show examples model predictions")       
argparser.add_argument("--weights_filepath", dest="weights_filepath", default=CONFIG['weights_filepath'],
                         help="Determines wether to show examples model predictions")                          

args = argparser.parse_args()


args.log_name = "logs/"+args.log_name+".txt"
args.datafile = "data/"+args.datafile

print("Data file name: ",args.datafile)

show_samples = int(args.show_samples)!=0
rand_embed = int(args.rand_embed)!=0
train_embed = int(args.train_embed)!=0
load_weights = int(args.load_weights)!=0

#Reading data file
data = pd.read_csv(args.datafile, sep="|", encoding="utf-8")

text = []
labels = []

#Read in Tokens, Labels from file
for x in range(len(data)):
    try:
        text.append(data.words[x].split("~"))
        labels.append(data["drug"][x])
    except Exception as e:
        print(x, e)
        continue
        
print("Sentences: ", len(text))

#Generate list of vocabulary
all_words = list(set(np.concatenate(text)))

#Getting maximum sequence length, necessary for padding
longest = 0
for sen in text:
    if len(sen)>longest:
        longest = len(sen)

print("Longest sentence: ", longest) 

#List of labels
print("Labels: ", len(labels))
all_labels = list(set(labels))

#Generate index of words
word_index = {w: i+1 for i, w in enumerate(all_words)}
word_index["~pad~"] = 0
all_words.insert(0,"~pad~")

#Generate index of labels
tag_index = {t: i for i, t in enumerate(all_labels)}

#Select top N words from vocabulary, filters out unusual words
top_words = args.top_words
num_words = len(word_index)
num_classes = len(tag_index)

print('Found %s unique tokens.' % len(word_index))

#Replace words with their index in data
X = [[word_index[w] for w in s] for s in text]
y = [tag_index[s] for s in labels]

print('Shape of label tensor pre-categorisation:', len(y))


#Pad input sequences
max_review_length = longest 
X = sequence.pad_sequences(X, maxlen=max_review_length, padding='post', value=0)

#Prepare label arrays for input into model
y = to_categorical(y, num_classes=num_classes)
print('Shape of label tensor post-categoridation:', y.shape)

print('Shape of data tensor:', X.shape)

#Load pre-trained embeddings file when random initialisation is not used
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
        
    embedding_matrix=[embedding_matrix]
    print("Embedding matrix created...")
else:
    #If random embeddings are used, manually set vector dimensions
    print("Random embeddings")
    embedding_matrix = None
    word_vectors_dim = 400

#Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)

if args.max_train is not None:
    max_train = int(args.max_train)
    if max_train<len(X_train):
        rain = X_train[:max_train]
        rain = y_train[:max_train]

        X_test = X_test[:(max_train*0.25)]
        y_test = y_test[:(max_train*0.25)]
	
print("Size of training set = ",len(X_train))

#Get configured optimiser
opt = args.optimizer.lower()
	
print("Creating Model...")

#Create input layer
input = Input(shape=(max_review_length,), name="input_drug_pred")



#Create Embedding Layer
model = Embedding(input_dim = num_words, output_dim = word_vectors_dim, 
                  input_length=max_review_length, weights = embedding_matrix, 
                  trainable=train_embed, mask_zero=True, 
                  name="embedding_drug_pred")(input)

#Load model parameters
#num_hidden = int(args.num_hidden)
dropout_rate = float(args.dropout_rate)
#Set batch size
if len(X_train)<int(args.batch_size):
	batch_size = len(X_train)
else:
	batch_size = int(args.batch_size)

#Create Bi-LSTM Layer
model = Bidirectional(LSTM(int(args.hidden_dim), 
                    dropout =dropout_rate, activation=args.lstm_act), 
                    name="lstm_layer")(model)


#Create Dense layer, will output one prediction per sentence
out = Dense(num_classes, activation=args.dense_act, name="dense_drug_pred")(model)

#Finalise model
model = Model(input,out)

#Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

#Prints a summary of model layers and parameters
#print(model.summary())

#Custom metric class
metrics = Metrics(tag_index=all_labels, batch_size=batch_size, log_name = args.log_name, train_data=(X_train,y_train))

#Configure location for saved weights, create checkpointing and early stop
filepath="temp/"+args.weights_filepath
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min', baseline=None)

#Train model
history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=int(args.num_epochs), verbose=1, 
					callbacks = [metrics,checkpoint],
					validation_data=(X_test, np.array(y_test)))