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
    
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)                 
    
fig = plt.figure()
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()

metrics = [val_loss,train_loss]

for rank, column in enumerate(metrics):    
    # Plot each line separately with its own color, using the Tableau 20    
    # color set in order.    
    plt.plot(N,column,lw=2.5, color=tableau20[rank])  

#ax.plot(N, f1 , '-b', label='F1')
#ax.plot(N, prec, '-r', label='Precision')
#ax.plot(N, rec, '-g', label='Recall')
#ax.plot(N, val_loss, '-b', label='Validation')
#ax.plot(N, train_loss, '-r', label='Training')
plt.xlabel('Train Dataset Size')
plt.ylabel('Loss')
plt.ylim((0,1))
ax.grid(color='grey', linestyle='--', linewidth=1, axis="y")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.savefig("figures/"+args.figname+".png")
plt.close()

print("Plot created at {}".format("figures/"+args.figname))
        
