import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data1 = pd.read_csv(
    '/home/yz2337/project/multi_fmri/results/percent/100/v5_pretrain_mlp/roc_data.csv')
data2 = pd.read_csv(
    '/home/yz2337/project/multi_fmri/results/percent/100/v4_pretrain_mlp/roc_data.csv')
data3 = pd.read_csv(
    '/home/yz2337/project/multi_fmri/results/percent/100/v3_pretrain_mlp/roc_data.csv')
data4 = pd.read_csv(
    '/home/yz2337/project/multi_fmri/results/percent/100/v0_nopretrain/roc_data.csv')

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(data1['FPR'], data1['TPR'], color='darkorange',
         lw=2, label=f'Mask all regions of random time period')
plt.plot(data2['FPR'], data2['TPR'], color='blue',
         lw=2, label=f'Mask all time period of random brain region')
plt.plot(data3['FPR'], data3['TPR'], color='purple',
         lw=2, label=f'Random masking')
plt.plot(data4['FPR'], data4['TPR'], color='darkgreen',
         lw=2, label=f'Scratch transformer')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/home/yz2337/project/multi_fmri/results/percent/100/roc.png')
