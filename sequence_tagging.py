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
from custom_tb_log import TrainValTensorBoard
from custom_metrics import Metrics
from custom_metrics import Metrics_Approx
from config import CONFIG
from ast import literal_eval
from nltk.corpus import stopwords
import string
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from custom_metrics import evaluate_approx_match
from sklearn.model_selection import KFold 
import datetime

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
argparser.add_argument("--show_samples", dest="show_samples", default=CONFIG['show_samples'],
                         help="Determines wether to show examples model predictions")       
argparser.add_argument("--weights_filepath", dest="weights_filepath", default=CONFIG['weights_filepath'],
                         help="Determines wether to show examples model predictions")                         

                         
                         
args = argparser.parse_args()

args.log_name = "logs/kfold/"+args.log_name+".txt"
args.datafile = "data/"+args.datafile

print("Data file name: ",args.datafile)

rand_embed = int(args.rand_embed)!=0
train_embed = int(args.train_embed)!=0
load_weights = int(args.load_weights)!=0
show_samples = int(args.show_samples)!=0

data = pd.read_csv(args.datafile, sep="|", encoding="utf-8")

norm_text_lit = []
labels_lit = []
pos_lit = []

stop_words = set(stopwords.words('english'))

for x in range(len(data)):
    try:
        #iobs = [z.replace("B","I").replace("M","O").replace("I-IND","O") for z in data["iob"][x].split("~")]
        iobs = [z.replace("B","I").replace("M","O") for z in data["iob"][x].split("~")]
        if "" in iobs:
            print(x,iobs)
        norm_text_lit.append(data.words[x].split("~"))
        labels_lit.append(iobs)
        pos_lit.append(data.words[x].split("~"))
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
#all_labels = list(set(np.concatenate(labels_lit)))
all_labels = ["I-ADR","I-IND","O"]

all_labels_full = list(np.concatenate(labels_lit))

labels_df = pd.DataFrame(all_labels_full)  

word_index = {w: i+1 for i, w in enumerate(all_words)}
word_index["~pad~"] = 0
all_words.insert(0,"~pad~")

all_labels.append("<PAD>")
tag_index = {t: i for i, t in enumerate(all_labels)}

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

#MULTICLASS
y = [to_categorical(i, num_classes=num_classes) for i in y]
y = np.array(y)

print('Shape of data tensor:', X.shape)

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

#if args.optimizer.lower()=="rmsprop":
	#opt = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=None, decay=0.0)
#else:
opt = "adam"
   
#class_weight = args.class_weights    
	
print("Begin cross validation")
kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
k = 0

with open(args.log_name, 'w', encoding='UTF-8') as writer:
    writer.write("timestamp|k|n|prec_a|rec_a|f1_a|val_loss|val_acc|train_loss|train_acc\n") 

for train_index, test_index in kf.split(X): 
    k+=1
    print("Starting K-",k)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
            
    if args.max_train is not None:
        max_train = int(args.max_train)
        if max_train<len(X_train):
            X_train = X_train[:max_train]
            y_train = y_train[:max_train]
            
            X_test = X_test[:66]
            y_test = y_test[:66]
        
    print("Size of training set = ",len(X_train))

    if len(X_train)<int(args.batch_size):
        batch_size = len(X_train)
    else:
        batch_size = int(args.batch_size)
        
    print("Creating Model...")

    if rand_embed:
        print("Random embeddings")
        embedding_matrix = None
        word_vectors_dim = 400

    dropout_rate = float(args.dropout_rate)
    num_hidden = int(args.num_hidden)

    input = Input(shape=(max_review_length,), name="input_layer")
    #Add the embedding layer	
    model = Embedding(input_dim = num_words, output_dim = word_vectors_dim, input_length=max_review_length, weights = embedding_matrix, trainable=train_embed, mask_zero=True, name="embedding_layer")(input)
    
    #Add Additional hidden LSTM layers
    #if num_hidden>1:
    #	for x in range(1, num_hidden):
            #model = Bidirectional(LSTM(int(args.hidden_dim),  
             #           return_sequences = True, 
              #          dropout =dropout_rate, activation=args.lstm_act))(model)
              
              
    #Add last hidden LSTM layer
    model = Bidirectional(LSTM(int(args.hidden_dim), 
                        return_sequences = True, 
                        dropout =dropout_rate, activation=args.lstm_act), name="lstm_layer")(model)
                        

    #Add final dense layer MULTICLASS
    out = TimeDistributed(Dense(num_classes, activation=args.dense_act ), name="dense_layer")(model)

    model = Model(input,out)
    
    if load_weights:
        #weights_filepath="temp/transfer_learning_weights_ds"
        print("Loading Weights")
        model.load_weights(args.weights_filepath, by_name=True) 

    #MULTICLASS
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    #print(model.summary())
    
    metrics = Metrics_Approx(tag_index=all_labels, k = k)


    history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=int(args.num_epochs), verbose=1,
                        callbacks = [metrics],
                        validation_data=(X_test, np.array(y_test)))
    
    y_pred = np.asarray(model.predict(X_test))
    y_pred = np.argmax(y_pred, axis=-1)

    targ = y_test
    targ = np.argmax(targ, axis=-1) 

    k_scores = evaluate_approx_match(y_pred, targ, all_labels)
    k_metrics_val = model.evaluate(X_test, y_test)
    k_metrics_train = model.evaluate(X_train, y_train)
    
    timestamp = str(datetime.datetime.now())
    
    with open(args.log_name, 'a', encoding='UTF-8') as writer:
        writer.write(timestamp + "|" + str(k)+ "|" + str(len(X_train)) + "|" + str(k_scores["p"])+ "|" + str(k_scores["r"])+ "|" + str(k_scores["f1"]) 
                        + "|" + str(k_metrics_val[0]) + "|" + str(k_metrics_val[1]) + "|" + str(k_metrics_train[0]) + "|" + str(k_metrics_train[1]) + "\n") 

    print("Best Scores for K-{}: Precision = {}, Recall = {}, F1 = {}".format(str(k), k_scores["p"], k_scores["r"],  k_scores["f1"]))

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
            writer.write("---------------------------------------\n")
            writer.write("---------------------------------------\n")

        
