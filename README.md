# adr-extraction
Author: Nestor Prieto Chavana

This repo contains the code used for my MSc research project:
> Extraction of Adverse Drug Reaction Reports from Social Media: A Deep Learning Approach

Since it is against Twitter terms of service to publish tweet text, the data has not been shared.

TODO: share tweet IDs and publish script to automate download.

#### Files & Directories

Content | Description 
--- | --- 
`sequence_tagging.py/` | Main model file. Loads text files containing post text and outputs label predictions for each token
`custom_metrics.py/` | Calculates approximate matching scores and performs custom logging, checkpoints and early stopping.
`drug_name_pred.py/` | Modified version of model to predict drug mentions from context. Used for transfer learning.
`sequence_tagging_predict_only.py/` | Modified version of sequence_tagging.py changed to only perform prediction.
`config.py/` | Contains default values for script parameters
`data/` | Place data files here
`logs/` | Location where log files will be saved
`temp/` | Location where checkpoint weights will be saved
`word2vec_models/` | Location to place pre-trained word2vec models
`custom_tb_log.py/` | Custom tensorboard logging. Unused in latest version.
`learning_curves.py/` | Generates learning plots for model. Unused in latest version.

As part of this research I supplemented the ASU and CHOP twitter datasets published in: 

> Cocos, A., Fiks, A.G. and Masino, A.J., 2017. Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts. Journal of the American Medical Informatics Association, 24(4), pp.813-821.

and

> Sarker A, Gonzalez G; Portable Automatic Text Classification for Adverse Drug Reaction Detection via Multi-corpus Training, Journal of Biomedical Informatics, 2015 Feb;53:196-207. doi: 10.1016/j.jbi.2014.11.002. Epub 2014 Nov 8. (resources for feature extraction for this task can be found at: 

The details to access these datasets can be found in: https://github.com/chop-dbhi/twitter-adr-blstm