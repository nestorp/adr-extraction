######
# Config.py
# 
# Author: Nestor Prieto Chavana
# Date: 10/8/2018
#
# Contains default settings for script parameters 
#
###
CONFIG = {
"datafile" : "full_tweets_adronly.csv",
"sentence_len" : 280,
"top_words" : 20000,
"word2vecpath" : "w2v_embeddings/word2vec_twitter_model.bin",
#"word2vecpath" : "w2v_embeddings/GoogleNews-vectors-negative300.bin",
"num_hidden" : 1,
"hidden_dim" : 256,
"lstm_act": "tanh",
"dense_act": "softmax",
"optimizer": "adam",
"learning_rate":0.001,
"class_weights" : None,
"train_embed" : 0,
"rand_embed" : 0,
"max_train" : None,
"seed" : None,
"num_epochs":6,
"batch_size":4,
"log_name" : "run1",
"dropout_rate" : 0.5,
"load_weights" : 0,
"show_samples" : 0,
"trans_learn" : 0,
"weights_filepath" : "temp/default_weights"
}