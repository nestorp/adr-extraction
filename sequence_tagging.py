######
# Sequence_tagging.py
# 
# Author: Nestor Prieto Chavana
# Date: 10/8/2018
#
# This script performs sequence tagging of ADRs in social media posts. 
# Files are expected to come in tab separated format, one line per post/tweet
# with the following columns: body|wordlen|words|pos|iob|adr
#
# Outputs predicted labels for each token in each sentence, which can be:
# I-ADR, I-IND, O, PAD
###

import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from argparse import ArgumentParser
from gensim.models import KeyedVectors
from custom_metrics import Metrics_Approx
from config import CONFIG
from custom_metrics import evaluate_approx_match
from sklearn.model_selection import KFold 
import datetime

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

rand_embed = int(args.rand_embed)!=0
train_embed = int(args.train_embed)!=0
load_weights = int(args.load_weights)!=0
show_samples = int(args.show_samples)!=0

#Reading data file
data = pd.read_csv(args.datafile, sep="|", encoding="utf-8")

text = []
labels = []
pos = []

#Read in Tokens, Labels and POS from file
for x in range(len(data)):
    try:
        iobs = [z.replace("B","I").replace("M","O") for z in data["iob"][x].split("~")]
        if "" in iobs:
            print(x,iobs)
        text.append(data.words[x].split("~"))
        labels.append(iobs)
        pos.append(data.words[x].split("~"))
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
#all_labels = list(set(np.concatenate(labels)))
all_labels = ["I-ADR","I-IND","O"]

#Generate index of words
word_index = {w: i+1 for i, w in enumerate(all_words)}
word_index["~pad~"] = 0
all_words.insert(0,"~pad~")

#Generate index of labels
all_labels.append("<PAD>")
tag_index = {t: i for i, t in enumerate(all_labels)}

#Select top N words from vocabulary, filters out unusual words
top_words = args.top_words
num_words = len(word_index)
num_classes = len(tag_index)
print('Found %s unique tokens.' % len(word_index))

#Replace words with their index in data
X = [[word_index[w] for w in s] for s in text]
y = [[tag_index[w] for w in s] for s in labels]

#Pad input sequences
max_review_length = longest 
X = sequence.pad_sequences(X, maxlen=max_review_length, padding='post', value=0)
y = sequence.pad_sequences(y, maxlen=max_review_length, padding='post', value=tag_index["<PAD>"])

#Prepare label arrays for input into model
y = [to_categorical(i, num_classes=num_classes) for i in y]
y = np.array(y)

print('Shape of data tensor:', X.shape)

#Load pre-trained embeddings file when random initialisation is not used
if not rand_embed:
    print("Loading w2v embeddings...")
    word2vecpath = args.word2vecpath
    word_vectors = KeyedVectors.load_word2vec_format(word2vecpath, binary=True, unicode_errors='replace')

    print("Finished loading w2v embeddings...")

    word_vectors_dim = len(np.asarray(word_vectors.get_vector(list(word_vectors.vocab.keys())[0]),dtype='float32'))
    print('Vector length:',word_vectors_dim)

    #Creating embedding index, will be used for initial weights in Embedding layer
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

#Get configuredo optimiser
opt = args.optimizer.lower()


#Beginning five fold cross validation
print("Begin cross validation")
kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
k = 0

#Print header into log file
with open(args.log_name, 'w', encoding='UTF-8') as writer:
    writer.write("timestamp|k|n|prec_a|rec_a|f1_a|val_loss|val_acc|train_loss|train_acc\n") 

for train_index, test_index in kf.split(X): 
    k+=1
    print("Starting K-",k)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
            
    #limit train and testing dataset size, useful for plotting learning curves
    if args.max_train is not None:
        max_train = int(args.max_train)
        if max_train<len(X_train):
            X_train = X_train[:max_train]
            y_train = y_train[:max_train]

            X_test = X_test[:(max_train*0.25)]
            y_test = y_test[:(max_train*0.25)]
        
    print("Size of training set = ",len(X_train))

    #Set batch size
    if len(X_train)<int(args.batch_size):
        batch_size = len(X_train)
    else:
        batch_size = int(args.batch_size)
        
    print("Creating Model...")

    #Read dropout rate
    dropout_rate = float(args.dropout_rate)
    #num_hidden = int(args.num_hidden)
    
    #Creating input layer
    input = Input(shape=(max_review_length,), name="input_layer")
    
    #Creating Embedding layer	
    model = Embedding(input_dim = num_words, output_dim = word_vectors_dim, 
                      input_length=max_review_length, weights = embedding_matrix, 
                      trainable=train_embed, mask_zero=True, name="embedding_layer")(input)
    

    #Add Bi-LSTM layer
    #Return sequence paremeter necessary for outputing prediction for each timestep
    model = Bidirectional(LSTM(int(args.hidden_dim), 
                        return_sequences = True, 
                        dropout =dropout_rate, activation=args.lstm_act), name="lstm_layer")(model)
                        

    #Add Time Distributed Dense layer
    #Time distributed outputs a prediction for each time step
    out = TimeDistributed(Dense(num_classes, activation=args.dense_act ), name="dense_layer")(model)

    #Finalise model
    model = Model(input,out)
    
    #Used for loading pre-saved model weights
    if load_weights:
        print("Loading Weights")
        model.load_weights(args.weights_filepath, by_name=True) 

    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    #Prints a summary of model layers and parameters
    #print(model.summary())
    
    #Custom metric calculations
    metrics = Metrics_Approx(tag_index=all_labels, k = k)

    #Train model 
    history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=int(args.num_epochs), verbose=1,
                        callbacks = [metrics],
                        validation_data=(X_test, np.array(y_test)))
    
    #Perform prediction to calculate scores for K iteration
    y_pred = np.asarray(model.predict(X_test))
    y_pred = np.argmax(y_pred, axis=-1)
    targ = y_test
    targ = np.argmax(targ, axis=-1) 
    k_scores = evaluate_approx_match(y_pred, targ, all_labels)
    k_metrics_val = model.evaluate(X_test, y_test)
    k_metrics_train = model.evaluate(X_train, y_train)
    
    #Get timestamp for log
    timestamp = str(datetime.datetime.now())
    
    #Print scores for current K iteration
    with open(args.log_name, 'a', encoding='UTF-8') as writer:
        writer.write(timestamp + "|" + str(k)+ "|" + str(len(X_train)) + "|" + str(k_scores["p"])+ "|" + str(k_scores["r"])+ "|" + str(k_scores["f1"]) 
                        + "|" + str(k_metrics_val[0]) + "|" + str(k_metrics_val[1]) + "|" + str(k_metrics_train[0]) + "|" + str(k_metrics_train[1]) + "\n") 

    print("Best Scores for K-{}: Precision = {}, Recall = {}, F1 = {}".format(str(k), k_scores["p"], k_scores["r"],  k_scores["f1"]))

#Print prediction samples for last iteration
if show_samples:
    with open("logs/samples/sample_" + args.datafile.replace("data/","").replace(".csv",".txt"), 'w', encoding='UTF-8') as writer:
        writer.write("{:25} ({:6}): {}\n".format("Word", "True", "Pred"))
        for i in range(len(X_test)):
            p = model.predict(np.array([X_test[i]]))
            p = np.argmax(p, axis=-1)
            for w, l, pred in zip(X_test[i], y_test[i], p[0]):
                if w>0:
                    l = np.argmax(l, axis=-1)
                    writer.write("{:25} {:6}:{}\n".format(all_words[w], all_labels[l], all_labels[pred]))
            writer.write("---------------------------------------\n")

        
