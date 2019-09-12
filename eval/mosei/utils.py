import numpy as np

def weighted_accuracy(t, p):
    tp = np.sum( np.logical_and( t,  p), axis=0, dtype=np.float32)
    tn = np.sum( np.logical_and(~t, ~p), axis=0, dtype=np.float32)

    pos = np.sum(t, axis=0, dtype=np.float32)
    neg = np.sum(~t, axis=0, dtype=np.float32)

    return (tp*neg/pos + tn) / (2*neg)

def reorder_labels(x) :
    # Orignial label order is happy,sad,anger,surprise,disgust,fear
    # Re-order alphabetically to Anger Disgust Fear Happy Sad Surprise
    return [ x[2], 
             x[4], 
             x[5], 
             x[0], 
             x[1], 
             x[3]] 

def reformat_array(x, fixed_point=3):                          
    s = [ f'{k:2.2f}' for k in x ]
    return ','.join(s)

