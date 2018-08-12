######
# Custom_metrics.py
# 
# Author: Nestor Prieto Chavana
# Date: 10/8/2018
#
# This script performs approximate matching calculations for model metrics
# and logs results 
#
###
import keras
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import datetime

def get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i):
    ''' Calculate approximate matching scores from predictions
    This function was adapted from the code published in 
    [Cocos, A., Fiks, A.G. and Masino, A.J., 2017. 
    Deep learning for pharmacovigilance: recurrent neural network 
    architectures for labeling adverse drug reactions in Twitter posts. 
    Journal of the American Medical Informatics Association] 
    under the GPL-3.0 license
    '''
    def find_inds(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def approx_match(l1,l2):
        return len([l for l in l1 if l in l2]) > 0

    # count tags and matches
    auto_tags_adr = 0
    auto_tags_indic = 0

    hand_tags_adr = 0
    hand_tags_indic = 0

    matches_adr = 0
    matches_indic = 0
    for a_seq,h_seq in zip(auto_seqs, hand_seqs):
               
        a_cntr_adr = [0] * len(a_seq)
        a_cntr_indic = [0] * len(a_seq)
        h_cntr_adr = [0] * len(h_seq)
        h_cntr_indic = [0] * len(h_seq)
        a_mask_adr = [0] * len(a_seq)
        a_mask_indic = [0] * len(a_seq)
        h_mask_adr = [0] * len(h_seq)
        h_mask_indic = [0] * len(h_seq)
        for i in range(1, len(a_seq)):

            if (a_seq[i-1] == o_i and adr_i == a_seq[i]):
                a_cntr_adr[i] = a_cntr_adr[i-1] + 1
            else:
                a_cntr_adr[i] = a_cntr_adr[i-1]
            if (a_seq[i-1] == o_i and ind_i == a_seq[i]):
                a_cntr_indic[i] = a_cntr_indic[i-1] + 1
            else:
                a_cntr_indic[i] = a_cntr_indic[i-1]

            if (h_seq[i-1] == o_i and adr_i == h_seq[i]):
                h_cntr_adr[i] = h_cntr_adr[i-1] + 1
            else:
                h_cntr_adr[i] = h_cntr_adr[i-1]
            if (h_seq[i-1] == o_i and ind_i == h_seq[i]):
                h_cntr_indic[i] = h_cntr_indic[i-1] + 1
            else:
                h_cntr_indic[i] = h_cntr_indic[i-1]

            a_mask_adr[i] = adr_i == a_seq[i]
            a_mask_indic[i] = ind_i == a_seq[i]
            h_mask_adr[i] = adr_i == h_seq[i]
            h_mask_indic[i] = ind_i == h_seq[i]
        a_cntr_adr = [a*m for a,m in zip(a_cntr_adr,a_mask_adr)]
        a_cntr_indic = [a*m for a,m in zip(a_cntr_indic,a_mask_indic)]

        h_cntr_adr = [h*m for h,m in zip(h_cntr_adr,h_mask_adr)]
        h_cntr_indic = [h*m for h,m in zip(h_cntr_indic,h_mask_indic)]

        auto_tags_adr += max(a_cntr_adr)
        auto_tags_indic += max(a_cntr_indic)
        hand_tags_adr += max(h_cntr_adr)
        hand_tags_indic += max(h_cntr_indic)
        a_subseqs_adr = [find_inds(a_cntr_adr,j) for j in range(1, max(a_cntr_adr)+1)]
        a_subseqs_indic = [find_inds(a_cntr_indic,j) for j in range(1, max(a_cntr_indic)+1)]
        h_subseqs_adr = [find_inds(h_cntr_adr,j) for j in range(1, max(h_cntr_adr)+1)]
        h_subseqs_indic = [find_inds(h_cntr_indic,j) for j in range(1, max(h_cntr_indic)+1)]
        
        matches_adr += sum([1 for a in a_subseqs_adr if sum([1 for h in h_subseqs_adr if approx_match(a,h)]) > 0])
        matches_indic += sum([1 for a in a_subseqs_indic if sum([1 for h in h_subseqs_indic if approx_match(a,h)]) > 0])

    try:
        precision_adr = float(matches_adr) / float(auto_tags_adr)
    except ZeroDivisionError:
        precision_adr = 0.0
    try:
        precision_indic = float(matches_indic) / float(auto_tags_indic)
    except ZeroDivisionError:
        precision_indic = 0.0
    try:
        recall_adr = float(matches_adr) / float(hand_tags_adr)
    except ZeroDivisionError:
        recall_adr = 0.0
    try:
        recall_indic = float(matches_indic) / float(hand_tags_indic)
    except ZeroDivisionError:
        recall_indic = 0.0
    try:
        f1score_adr = 2*precision_adr*recall_adr/(precision_adr + recall_adr)
    except ZeroDivisionError:
        f1score_adr = 0.0
    try:
        f1score_indic = 2*precision_indic*recall_indic/(precision_indic+recall_indic)
    except ZeroDivisionError:
        f1score_indic = 0.0
    try:
        precision = float(matches_adr + matches_indic) / float(auto_tags_adr + auto_tags_indic)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(matches_adr + matches_indic) / float(hand_tags_adr + hand_tags_indic)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f1score = 0.0
    #print('Approximate Matching Results:\n  ADR: Precision '+ str(precision_adr)+ ' Recall ' + str(recall_adr) + ' F1 ' + str(f1score_adr)
        #+ '\n  Indication: Precision ' + str(precision_indic) + ' Recall ' + str(recall_indic) + ' F1 ' + str(f1score_indic)
        #+ '\n  Overall: Precision ' + str(precision) + ' Recall ' + str(recall) + ' F1 ' +  str(f1score))
    return {'p':precision_adr, 'r':recall_adr, 'f1':f1score_adr}
    
class Metrics_Approx(keras.callbacks.Callback):
    ''' Performs custom logging functions, approximate matching and 
    exact matching scores for each iteration of the model. It keeps track
    of model performance and saves a checkpoint every time there is improvement.
    Additionally, it provides early stopping functionality.
    '''
    def __init__(self, tag_index = None, k = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.tag_index = tag_index
        self.k = k
        self.scores = {}
        self.best_epoch = None
        self.no_improv_epoch = 0
        self.tol = 2
    
    #Intreface for the get_approx_match function
    def get_approx_match_wrapper(self, auto_seqs, hand_seqs, adr_i, o_i,ind_i):
        return get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i)
        
    #Save weights model weights to file
    def save_model_weights(self):
        self.model.save_weights(self.weights_filepath)
    
   
    #At end of each epoch, log scores and save weights if an improvement 
    #was produced
    def on_epoch_end(self, epoch, logs={}):
        
        self.weights_filepath="temp/seq_tag_weights_best_k" + str(self.k)
        
        timestamp = str(datetime.datetime.now())

        tag_index = self.tag_index
        
        for i in range(len(tag_index)):
            if tag_index[i]=="I-ADR":
                adr_i = i
            if tag_index[i]=="I-IND":
                ind_i = i
            if tag_index[i]=="O":
                o_i = i
    
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.argmax(predict, axis=-1)

        targ = self.validation_data[1]
        targ = np.argmax(targ, axis=-1) 
        
        approx_scores = self.get_approx_match_wrapper(predict, targ, adr_i, o_i, ind_i)      
        
        predict = np.concatenate(predict)
        targ = np.concatenate(targ)

        precision_e=precision_score(targ, predict, average=None)
        recall_e=recall_score(targ, predict, average=None)
        f1_e=f1_score(targ, predict, average=None)
        
        #Save epoch scores
        with open(self.log_filename, 'a', encoding='UTF-8') as writer:
            writer.write(timestamp + "|" + str(epoch+1) + "|" + str(precision_e[adr_i]) + "|" + str(recall_e[adr_i]) + "|" + str(f1_e[adr_i]) + "|" + str(approx_scores["p"]) + "|" + str(approx_scores["r"]) + "|" + str(approx_scores["f1"]) + "\n")
        
        print("Exact Match Scores Class ADR: Precision = {}, Recall = {}, F1 = {}".format(precision_e[adr_i], recall_e[adr_i],  f1_e[adr_i]))
        print("Approximate Matching Scores Class ADR: Precision = {}, Recall = {}, F1 = {}".format(approx_scores["p"], approx_scores["r"],  approx_scores["f1"]))
        
        #Determine if improvement has been produced, and if so save current 
        #weights. If no improvement has been found after a number of 
        #iterations > tol, stop training
        self.scores[epoch] = approx_scores
        if epoch==0:
            print("Epoch {}: F1 improved from inf to {}, saving model to {}".format(epoch+1, self.scores[epoch]["f1"], self.weights_filepath))
            self.best_epoch = epoch
            self.save_model_weights()
        if epoch>0:
            if self.scores[epoch]["f1"] > self.scores[self.best_epoch]["f1"]:
                print("Epoch {}: F1 improved from {} to {}, saving model to {}".format(epoch+1, self.scores[self.best_epoch]["f1"], self.scores[epoch]["f1"], self.weights_filepath))
                self.best_epoch = epoch
                self.save_model_weights()
            else:
                print("Epoch {}: F1 did not improve from {}".format(epoch+1, self.scores[self.best_epoch]["f1"], self.scores[epoch]["f1"], self.weights_filepath))
            
            if self.scores[epoch]["f1"] > self.scores[epoch-1]["f1"]:
                self.no_improv_epoch = 0
            else:
                self.no_improv_epoch+=1
                if self.no_improv_epoch>self.tol:
                    print("Epoch {}: early stopping".format(epoch+1))
                    self.model.stop_training = True
        
        return
        
    #Create log file and print header
    def on_train_begin(self, logs={}):
        timestamp = str(datetime.datetime.now())
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")
        
        filename = "logs/a_metrics_" + timestamp_str + ".txt"
        
        self.log_filename = filename
        
        with open(self.log_filename, 'w', encoding='UTF-8') as writer:
            writer.write("timestamp|epoch|prec_e|rec_e|f1_e|prec_a|rec_a|f1_a\n")
        
        return
        
    #Restore best performing weights on model
    def on_train_end(self, logs={}):
        print("Restoring best weights")
        self.model.load_weights(self.weights_filepath)        
        return

       
        
class Metrics(keras.callbacks.Callback):
    ''' Performs custom logging functions using exact matching scores for each 
    iteration of the model.
    '''
    def __init__(self, tag_index = None, batch_size = 0, log_name=None, train_data=None):
        super(keras.callbacks.Callback, self).__init__()
        self.tag_index = tag_index
        self.batch_size=batch_size
        self.scores = {}
        self.log_filename=log_name
        self.training_data = train_data


    #At end of each epoch, log exact matching scores
    def on_epoch_end(self, epoch, logs={}):
    
        timestamp = str(datetime.datetime.now())
           
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.argmax(predict, axis=-1)

        targ = self.validation_data[1]
        targ = np.argmax(targ, axis=-1)      

        precision_e=precision_score(targ, predict, average="weighted")
        recall_e=recall_score(targ, predict, average="weighted")
        f1_e=f1_score(targ, predict, average="weighted")
        
        val_scores = self.model.evaluate(self.validation_data[0],self.validation_data[1])

        
        train_scores = self.model.evaluate(self.training_data[0],self.training_data[1])
        
        
        with open(self.log_filename, 'a', encoding='UTF-8') as writer:
            writer.write(timestamp + "|" + str(epoch) + "|" + str(precision_e) + "|" + str(recall_e) + "|" + str(f1_e)  
            + "|" + str(val_scores[0]) + "|" + str(val_scores[1]) + "|" + str(train_scores[0]) + "|" + str(train_scores[1]) + "\n")

        print("Exact Match Scores Class: Precision = {}, Recall = {}, F1 = {}".format(precision_e, recall_e,  f1_e))
        return
        
    #Create log file at the start of training
    def on_train_begin(self, logs={}):
        timestamp = str(datetime.datetime.now())
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")
        
        
        if self.log_filename == None or self.log_filename == "run1":
            filename = "logs/drugname_batchsize"+str(self.batch_size)+"_metrics_" + timestamp_str + ".txt"
            self.log_filename = filename
        else:
            self.log_filename = "logs/"+self.log_filename+".txt"
        
        with open(self.log_filename, 'w', encoding='UTF-8') as writer:
            writer.write("timestamp|epoch|prec_e|rec_e|f1_e|val_loss|val_acc|train_loss|train_acc\n")
        
        return
        
     
#Interface to directly execute function get_approx_matching
def evaluate_approx_match(auto_seqs, hand_seqs, tag_index):
    for i in range(len(tag_index)):
        if tag_index[i]=="I-ADR":
            adr_i = i
        if tag_index[i]=="I-IND":
            ind_i = i
        if tag_index[i]=="O":
            o_i = i

    return get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i)