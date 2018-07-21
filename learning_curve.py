from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

argparser = ArgumentParser()

argparser.add_argument("--filelist", dest="filelist", help="List of filenames with data to be plotted")
argparser.add_argument("--figname", dest="figname", help="Name of file to export plot")

args = argparser.parse_args()

files = args.filelist.split(",")

rec = []
prec = []
f1 = []
N = []
val_loss=[]
train_loss=[]
val_acc=[]
train_acc=[]

for f in files:
    filename = "logs/kfold/" + f.strip() + ".txt"
    rec_a = []
    prec_a = []
    f1_a = []
    N_ = []
    val_loss_=[]
    train_loss_=[]
    val_acc_=[]
    train_acc_=[]
    with open(filename) as instream:
        header = instream.readline()
        headers = header.replace("\n","").split("|")
        for i in range(len(headers)):
            if headers[i] == "n":
                n_i = i
            if headers[i] == "f1_a":
                f_i = i
            if headers[i] == "rec_a":
                r_i = i
            if headers[i] == "prec_a":
                p_i = i
            if headers[i] == "val_loss":
                vl_i = i
            if headers[i] == "train_loss":
                tl_i = i
            if headers[i] == "val_acc":
                va_i = i
            if headers[i] == "train_acc":
                ta_i = i
            
        for line in instream:
            row = line.replace("\n","").split("|")
            rec_a.append(float(row[r_i].strip()))
            prec_a.append(float(row[p_i].strip()))
            f1_a.append(float(row[f_i].strip()))
            N_.append(int(row[n_i].strip()))
            val_loss_.append(float(row[vl_i].strip()))
            train_loss_.append(float(row[tl_i].strip()))
            val_acc_.append(float(row[va_i].strip()))
            train_acc_.append(float(row[ta_i].strip()))
            
    rec.append(np.mean(rec_a))
    prec.append(np.mean(prec_a))
    f1.append(np.mean(f1_a))
    N.append(np.max(N_))
    val_loss.append(np.mean(val_loss_))
    train_loss.append(np.mean(train_loss_))
    val_acc.append(np.mean(val_acc_))
    train_acc.append(np.mean(train_acc_))
    
fig = plt.figure()
ax = plt.subplot(111)
#ax.plot(N, f1 , '-b', label='F1')
#ax.plot(N, prec, '-r', label='Precision')
#ax.plot(N, rec, '-g', label='Recall')
ax.plot(N, val_loss, '-b', label='Validation')
ax.plot(N, train_loss, '-r', label='Training')
plt.xlabel('Train Dataset Size')
plt.ylabel('Loss')
plt.ylim((0,1))
ax.grid(color='grey', linestyle='-', linewidth=1, axis="y")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.savefig("figures/"+args.figname)
plt.close()

print("Plot created at {}".format("figures/"+args.figname))
        
