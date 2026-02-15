#This module implements Multifractal Analysis on a 1D discrete time series (token surprise scores). Its primary objective is to extract high-dimensional geometric features—Self-Similarity and Long-Range Dependence to create a statistical "fingerprint" of the input text.
import numpy as np
import json
#importing library for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress

scale = [8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320]
def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data
def generate_trajectory(RawScores):
    scores=np.array(RawScores)
    mu=np.mean(scores)
#calc standard deviation
    sigma=np.std(scores)
    increment=(scores-mu)/sigma
    #below line of code creates the random walk using the cumsum of increment
    trajectory=np.cumsum(increment)
    return trajectory,increment

def calc_S(x, scale_param=None, epsilon=0.1):
    # Use global scale if no scale is provided to avoid iteration errors
    current_scale = scale_param if scale_param is not None else scale
    
    #this is the various tocken window sizes we will use to calc S
    prob=[]
    for tau in current_scale:
        #important concept called array slicing used must delve deeper
        disp=np.abs(x[tau:]-x[:-tau])
        #Calculate what fraction of displacements are 'small' (near zero)
        p_tau=np.mean(disp<epsilon)
        prob.append(p_tau + 1e-10)
    #this will convert the scale we sampled to the x axis  
    log_tau = np.log(current_scale)
    #this will convert the probs we have to the y axis
    log_probs = np.log(prob)
    #pplying linear reggresion
    slope, intercept, r_value, p_value, std_err = linregress(log_tau, log_probs)
    s=-slope
    #Coefficient of Determination) tells you how well your data actually fits the mathematical model you are trying to use.
    #R^2 represents the proportion of the variance in the dependent variable
    #t tells you how much the text actually behaves like a fractal.
    return s, r_value**2 



def calc_H(increment):
    avg_rs_values = []
    for n in scale:
        chunks=len(increment)//n
        rs_ratio=[]
        for i in range(chunks):
            #divide an article into smaller chunks
            chk=increment[i*n:(i+1)*n]
            z = np.cumsum(chk - np.mean(chk))
            R = np.max(z) - np.min(z)
            S = np.std(chk)
            if S > 0:
                rs_ratio.append(R / S)
        if len(rs_ratio) > 0:
            avg_rs_values.append(np.mean(rs_ratio))
            
    log_n = np.log(scale[:len(avg_rs_values)])
    log_rs = np.log(avg_rs_values)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rs)
    H = slope
    return H, r_value**2

def calc_cosine(x,y):
    #this step converts the list into a [1,n] ordered matrix
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)
    #the below function automatically normalizes the vectors.
    return cosine_similarity(x,y)[0][0]

#now we analyse semantic drift, embeddings is the dimensions of a particular item 
# window size is set to find and avererage value to compare text
def analyse_semanticDrift(embeddings, window_size=50):
    num_block= len(embeddings)//window_size
    block=[]
    for i in range(num_block):
        # this process is called Temporal Pooling or Semantic Chunking
        #we slice the embeddings into blocks of size window_size and calculate the mean embedding for each block. 
        #This mean embedding represents the "average" semantic content of that block of text.
        block_slice=embeddings[i*window_size:(i+1)*window_size]
        block.append(np.mean(block_slice, axis=0))
        
    if len(block)<2:
        print("Not enough blocks to analyze semantic drift.")
        return None, None
        
    simm=[]
    for i in range(len(block)-1):
        sim=calc_cosine(block[i],block[i+1])
        simm.append(sim)
    return np.mean(simm), np.var(simm)

def get_allMetric(scores, embedding=None):
    trajectory,increment= generate_trajectory(scores)
    H, H_r2 = calc_H(increment)
    S, S_r2 = calc_S(trajectory, scale_param=scale)
    results = {
        "Hurst_Exp": float(H),
        "Hurst_R2": float(H_r2),
        "Holder_Exp": float(S),
        "Holder_R2": float(S_r2)
    }
    if embedding is not None:
        sem_mean, sem_var = analyse_semanticDrift(embedding)
        results["Semantic_Consistency"] = float(sem_mean)
        results["Semantic_Volatility"] = float(sem_var)
    return results
if __name__ == "__main__":
    data_list = load_json('fractal_scores.json')
    
    all_results = []
    for entry in data_list:
        sample_id = entry.get('id', 'unknown_id')
        scores = entry.get('scores', [])
        metrics = get_allMetric(scores, embedding=None)
        entry_result = {"id": sample_id, "metrics": metrics}
        all_results.append(entry_result)
        print(f"Processed: {sample_id}")
    with open('analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSuccessfully processed {len(all_results)} entries.")