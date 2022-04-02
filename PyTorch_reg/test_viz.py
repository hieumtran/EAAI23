import matplotlib.pyplot as plt

def read_output(path):
    text= open(path, 'r').read()
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
    
    viz(loss[:50], 'Euclidean', 'loss', './PyTorch_reg/figure/ResNet50_')
    viz(rmse_val[:50], 'rmse_val', 'rmse', './PyTorch_reg/figure/ResNet50_')
    viz(rmse_ars[:50], 'rmse_ars', 'rmse', './PyTorch_reg/figure/ResNet50_')
    viz(corr_val[:50], 'corr_val', 'corr', './PyTorch_reg/figure/ResNet50_')
    viz(corr_ars[:50], 'corr_ars', 'corr', './PyTorch_reg/figure/ResNet50_')
    viz(ccc_val[:50], 'ccc_val', 'ccc', './PyTorch_reg/figure/ResNet50_')
    viz(ccc_ars[:50], 'ccc_ars', 'ccc', './PyTorch_reg/figure/ResNet50_')
    viz(sagr_val[:50], 'sagr_val', 'sagr', './PyTorch_reg/figure/ResNet50_')
    viz(sagr_ars[:50], 'sagr_ars', 'sagr', './PyTorch_reg/figure/ResNet50_')
    

def viz(arr, title, yaxis, path):
    plt.figure(figsize=(15,7))
    plt.plot(arr)
    plt.ylabel(yaxis)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.grid()
    plt.savefig(path+title+'.png', dpi=500)
    plt.show()
    

read_output('./PyTorch_reg/log/resnet50_30_test.txt')

