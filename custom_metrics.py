from keras import backend as K
import keras
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle   
import datetime

def get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i):
    ''' Calculate approximate match from results file, written with format:
    word <actualLabel> <predictedLabel>

    Sentences should be demarcated by 'BOS' and 'EOS' lines.
    Labels can be one of 'O', 'B-ADR', 'I-ADR', 'B-Indication', 'I-Indication'

    :param filename: Name of file with results
    :return:
    '''
    def find_inds(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def approx_match(l1,l2):
        return len([l for l in l1 if l in l2]) > 0

    #auto_seqs = pr
    #hand_seqs = ta

    # count tags and matches
    auto_tags_adr = 0
    auto_tags_indic = 0

    hand_tags_adr = 0
    hand_tags_indic = 0

    matches_adr = 0
    matches_indic = 0
    for a_seq,h_seq in zip(auto_seqs, hand_seqs):
        
        #print(a_seq,h_seq)
        
        a_cntr_adr = [0] * len(a_seq)
        a_cntr_indic = [0] * len(a_seq)
        h_cntr_adr = [0] * len(h_seq)
        h_cntr_indic = [0] * len(h_seq)
        a_mask_adr = [0] * len(a_seq)
        a_mask_indic = [0] * len(a_seq)
        h_mask_adr = [0] * len(h_seq)
        h_mask_indic = [0] * len(h_seq)
        for i in range(1, len(a_seq)):

            if (a_seq[i-1] == o_i and adr_i == a_seq[i]):# or a_seq[i] == 'B-ADR':
                a_cntr_adr[i] = a_cntr_adr[i-1] + 1
            else:
                a_cntr_adr[i] = a_cntr_adr[i-1]
            if (a_seq[i-1] == o_i and ind_i == a_seq[i]):# or a_seq[i] == 'B-Indication':
                a_cntr_indic[i] = a_cntr_indic[i-1] + 1
            else:
                a_cntr_indic[i] = a_cntr_indic[i-1]

            if (h_seq[i-1] == o_i and adr_i == h_seq[i]):# or h_seq[i] == 'B-ADR':
                h_cntr_adr[i] = h_cntr_adr[i-1] + 1
            else:
                h_cntr_adr[i] = h_cntr_adr[i-1]
            if (h_seq[i-1] == o_i and ind_i == h_seq[i]):# or h_seq[i] == 'B-Indication':
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

    def __init__(self, tag_index = None, k = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.tag_index = tag_index
        self.k = k
        self.scores = {}
        self.best_epoch = None
        self.no_improv_epoch = 0
        self.tol = 2
    
    def get_approx_match_wrapper(self, auto_seqs, hand_seqs, adr_i, o_i,ind_i):
        return get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i)
        
    def save_model_weights(self):
        self.model.save_weights(self.weights_filepath)
    
   
    def on_epoch_end(self, epoch, logs={}):
        
        self.weights_filepath="temp/seq_tag_weights_best_k" + str(self.k)
        
        timestamp = str(datetime.datetime.now())
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")

        tag_index = self.tag_index#pickle.load(open("temp/tag_index", 'rb'))
        
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
        
        with open(self.log_filename, 'a', encoding='UTF-8') as writer:
            writer.write(timestamp + "|" + str(epoch+1) + "|" + str(precision_e[adr_i]) + "|" + str(recall_e[adr_i]) + "|" + str(f1_e[adr_i]) + "|" + str(approx_scores["p"]) + "|" + str(approx_scores["r"]) + "|" + str(approx_scores["f1"]) + "\n")
            #writer.write(str(approx_scores["p"]) + "|" + str(approx_scores["r"]) + "|" + str(approx_scores["f1"]) + "\n")
        
        print("Exact Match Scores Class ADR: Precision = {}, Recall = {}, F1 = {}".format(precision_e[adr_i], recall_e[adr_i],  f1_e[adr_i]))
        print("Approximate Matching Scores Class ADR: Precision = {}, Recall = {}, F1 = {}".format(approx_scores["p"], approx_scores["r"],  approx_scores["f1"]))
        
        self.scores[epoch] = approx_scores
        if epoch==0:
            print("Epoch {}: F1 improved from inf to {}, saving model to {}".format(epoch+1, self.scores[epoch]["f1"], self.weights_filepath))
            self.best_epoch = epoch
            self.save_model_weights()
        if epoch>0:
            if self.scores[epoch]["f1"] > self.scores[self.best_epoch]["f1"]:
                print("Epoch {}: F1 improved from {} to {}, saving model to {}".format(epoch+1, self.scores[epoch-1]["f1"], self.scores[epoch]["f1"], self.weights_filepath))
                self.best_epoch = epoch
                self.save_model_weights()
            else:
                print("Epoch {}: F1 did not improve from {}".format(epoch+1, self.scores[epoch-1]["f1"], self.scores[epoch]["f1"], self.weights_filepath))
            
            if self.scores[epoch]["f1"] > self.scores[epoch-1]["f1"]:
                self.no_improv_epoch = 0
            else:
                self.no_improv_epoch+=1
                if self.no_improv_epoch>self.tol:
                    print("Epoch {}: early stopping".format(epoch+1))
                    self.model.stop_training = True
        
        return
        
    def on_train_begin(self, logs={}):
        timestamp = str(datetime.datetime.now())
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")
        
        filename = "logs/a_metrics_" + timestamp_str + ".txt"
        
        self.log_filename = filename
        
        with open(self.log_filename, 'a', encoding='UTF-8') as writer:
            writer.write("timestamp|epoch|prec_e|rec_e|f1_e|prec_a|rec_a|f1_a\n")
        
        return
        
    def on_train_end(self, logs={}):
        print("Restoring best weights")
        self.model.load_weights(self.weights_filepath)        
        return

       
        
class Metrics(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
    
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")
           
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.argmax(predict, axis=-1)

        targ = self.validation_data[1]
        targ = np.argmax(targ, axis=-1)      
        
        #predict = np.concatenate(predict)
        #targ = np.concatenate(targ)

        precision_e=precision_score(targ, predict, average="weighted")
        recall_e=recall_score(targ, predict, average="weighted")
        f1_e=f1_score(targ, predict, average="weighted")
        
        with open('logs/e_metrics.txt', 'a', encoding='UTF-8') as writer:
            writer.write(timestamp + "|" + str(epoch) + "|" + str(precision_e) + "|" + str(recall_e) + "|" + str(f1_e) + "\n")
            #writer.write(str(approx_scores["p"]) + "|" + str(approx_scores["r"]) + "|" + str(approx_scores["f1"]) + "\n")

        #for x in range(len(precision_e)):
            #print("Exact Match Scores Class {}: Precision = {}, Recall = {}, F1 = {}".format(x, precision_e[x], recall_e[x],  f1_e[x]))
        print("Exact Match Scores Class: Precision = {}, Recall = {}, F1 = {}".format(precision_e, recall_e,  f1_e))
        return
        
    def on_train_begin(self, logs={}):
        timestamp = str(datetime.datetime.now())
        timestamp_str = timestamp.replace(" ","").replace(":","").replace(".","").replace("-","")
        
        filename = "logs/a_metrics_" + timestamp_str + ".txt"
        
        self.log_filename = filename
        
        return
        with open(self.log_filename, 'a', encoding='UTF-8') as writer:
            writer.write("timestamp|epoch|prec_e|rec_e|f1_e\n")
            
def evaluate_approx_match(auto_seqs, hand_seqs, tag_index):
    for i in range(len(tag_index)):
        if tag_index[i]=="I-ADR":
            adr_i = i
        if tag_index[i]=="I-IND":
            ind_i = i
        if tag_index[i]=="O":
            o_i = i

    return get_approx_match(auto_seqs, hand_seqs, adr_i, o_i,ind_i)