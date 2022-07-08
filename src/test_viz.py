import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def read_output(read_path, save_loc, save_name):
    text= open(read_path, 'r').read()
    text = text.split('\n')
    
    # Series of metrics
    loss = []
    rmse_val, rmse_ars = [], []
    corr_val, corr_ars = [], []
    sagr_val, sagr_ars = [], []
    ccc_val, ccc_ars = [], []
    for i in range(len(text)):
        if 'Test: ' in text[i]:
            tmp_text = text[i].split(' ')
            loss.append(float(tmp_text[1]))
            rmse_val.append(float(tmp_text[4])), rmse_ars.append(float(tmp_text[7]))
            corr_val.append(float(tmp_text[10])), corr_ars.append(float(tmp_text[13]))
            ccc_val.append(float(tmp_text[16])), ccc_ars.append(float(tmp_text[19]))
            sagr_val.append(float(tmp_text[22])), sagr_ars.append(float(tmp_text[25]))
        elif 'Val' in text[i]:
            tmp_text = text[i].split(' ')
            loss.append(float(tmp_text[2]))
            rmse_val.append(float(tmp_text[5])), rmse_ars.append(float(tmp_text[8]))
            corr_val.append(float(tmp_text[11])), corr_ars.append(float(tmp_text[14]))
            ccc_val.append(float(tmp_text[17])), ccc_ars.append(float(tmp_text[20]))
            sagr_val.append(float(tmp_text[23])), sagr_ars.append(float(tmp_text[26]))

    
    if os.path.isdir(save_loc) != True: os.mkdir(save_loc)
    
    viz(loss[:], 'Euclidean', 'loss', save_loc + save_name + '_')
    viz(rmse_val[:], 'rmse_val', 'rmse', save_loc + save_name + '_')
    viz(rmse_ars[:], 'rmse_ars', 'rmse', save_loc + save_name + '_')
    viz(corr_val[:], 'corr_val', 'corr', save_loc + save_name + '_')
    viz(corr_ars[:], 'corr_ars', 'corr', save_loc + save_name + '_')
    viz(ccc_val[:], 'ccc_val', 'ccc', save_loc + save_name + '_')
    viz(ccc_ars[:], 'ccc_ars', 'ccc', save_loc + save_name + '_')
    viz(sagr_val[:], 'sagr_val', 'sagr', save_loc + save_name + '_')
    viz(sagr_ars[:], 'sagr_ars', 'sagr', save_loc + save_name + '_')
    
    best_perf = loss.index(min(loss))
    worst_perf = loss.index(max(loss))
    output = 'Epoch {} | L2 distance: {} | RMSE_Val: {} | RMSE_Ars: {} |' \
            'P_Val: {} | P_Ars: {} | C_Val: {} | C_Ars: {} | S_Val: {} | S_Ars: {}'
            
    print('Best performance ' + output.format(best_perf, loss[best_perf], 
                                              rmse_val[best_perf], rmse_ars[best_perf],
                                              corr_val[best_perf], corr_ars[best_perf],
                                              ccc_val[best_perf], ccc_ars[best_perf], 
                                              sagr_val[best_perf], sagr_ars[best_perf]))
    
    print('Worst performance ' + output.format(worst_perf, loss[worst_perf], 
                                              rmse_val[worst_perf], rmse_ars[worst_perf],
                                              corr_val[worst_perf], corr_ars[worst_perf],
                                              ccc_val[worst_perf], ccc_ars[worst_perf], 
                                              sagr_val[worst_perf], sagr_ars[worst_perf]))
    

def viz(arr, title, yaxis, path):
    plt.figure(figsize=(15,7))
    plt.plot(np.arange(1, len(arr)+1, 1), arr, 'o-')
    
    # X tick
    plt.xticks(np.arange(1, len(arr)+1, 5))
    
    # Title and axis label
    plt.ylabel(yaxis)
    plt.xlabel('Epoch')
    plt.title(title)
    
    # Save configuration
    plt.tight_layout()
    plt.grid()
    plt.savefig(path+title+'.jpg', dpi=500)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_path", help="test log files")
    parser.add_argument("--save_loc", help="log file viz save location")
    parser.add_argument("--save_name", help="log file viz save name")
    args = parser.parse_args()
    read_output(args.read_path, args.save_loc, args.save_name)

