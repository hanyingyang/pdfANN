# pdfANN/postprocessing
from __future__ import annotations
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt

from data.pdfCZ import PdfMapConfig, pdfMap
from func import JSDiv, betaPDF

#---- Load PDF --------------------------------------------------------
dataSet = pdfMap(
        bins=[46, 56],
        mapID="hist",
        pdfType="marginal",
        timestep=[9],
        band=4,
        jump=18,
        filQty=True,
    )
dataSet.set_axis()

inputs = np.load("./data/Input_1snapshots_from_801_to_801_fs_9_jump_18.npy")
targets = np.load("./data/Target_hist_1snapshots_from_801_to_801_fs_9_jump_18.npy")
predictions = np.load("./Models/training_20251130-101011/prediction.npy")

Zst = 0.03
Z = np.exp(inputs[:,2]) * Zst

nc = 46; nZ = 56
pdf_c_dns = targets[:, :nc-1]; pdf_Z_dns = targets[:, nc-1:]
pdf_c_dns = pdf_c_dns / dataSet.cWidth_hist[None, :]
pdf_Z_dns = pdf_Z_dns / dataSet.lnZWidth_hist[None, :] / dataSet.ZCenter_hist[None, :]

pdf_c_ann = predictions[:, :nc-1]; pdf_Z_ann = predictions[:, nc-1:]
pdf_c_ann = pdf_c_ann / dataSet.cWidth_hist[None, :]
pdf_Z_ann = pdf_Z_ann / dataSet.lnZWidth_hist[None, :] / dataSet.ZCenter_hist[None, :]

pdf_c_beta = betaPDF(inputs[:,0], inputs[:,1], dataSet.cCenter_hist, dataSet.cWidth_hist)
pdf_Z_beta = betaPDF(Z, inputs[:, 3], dataSet.ZCenter_hist, dataSet.ZWidth_hist)

#--- Load reaction rate -----------------------------------------------------
# Doubly conditional averaging (reaction rate/density)
mat = loadmat('./data/conAvg_WcY_H2_Lifted_46c_56Z')
W_avg = mat['W_avg']
W_avg[np.isnan(W_avg)] = 0
W_avg = W_avg[:-1,:-1]

# Filtered reaction rate and density
filData = np.load('./data/Filtered_data_1snapshots_from_801_to_801_fs_9_jump_18.npy')
wBar = filData[:,0]
rhoBar = filData[:,1]

#---- Plot of marginal PDF ---------------------------------------------------
"""
Progress variable
"""
point = 3287

fig = plt.figure(figsize=(6,4))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.plot(dataSet.cCenter_hist,pdf_c_dns[point,:],label='DNS',color = 'black',linewidth=3)
ax.plot(dataSet.cCenter_hist,pdf_c_ann[point,:],label='ANN',color = 'blue',linewidth=3)
ax.plot(dataSet.cCenter_hist,pdf_c_beta[point,:],label='beta',color = 'orange',linewidth=3)

ax.set_xlabel('c', fontsize=22)
ax.set_ylabel('P(c)', fontsize=22)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.yticks(fontsize=20)
eq = '$\widetilde{c}$: %.4f \n$\widetilde{\sigma_c^2}$: %.6f' %(
                                             inputs[point, 0],
                                             inputs[point, 1] * (inputs[point, 0] * (1 - inputs[point, 0])))
ax.text(0.1,1,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)

"""
Mixture fraction
"""
Z = np.exp(inputs[point, 2]) * Zst
Zvar = inputs[point, 3] * (Z * (1 - Z))

fig = plt.figure(figsize=(6,4))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.plot(dataSet.ZCenter_hist[:20],pdf_Z_dns[point,:20],label='DNS',color = 'black',linewidth=3)
ax.plot(dataSet.ZCenter_hist[:20],pdf_Z_ann[point,:20],label='ANN',color = 'blue',linewidth=3)
ax.plot(dataSet.ZCenter_hist[:20],pdf_Z_beta[point,:20],label='beta',color = 'orange',linewidth=3)

ax.set_xlabel('Z', fontsize=22)
ax.set_ylabel('P(Z)', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
eq = '$\widetilde{Z}$: %.4f \n$\widetilde{\sigma}_Z^2$: %.6f' %(
                                             Z, Zvar)
ax.text(0.06,50,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)

#---- Plot of PDF of JSD -----------------------------------------------------------
JSD_c_ann = np.zeros(len(inputs))
JSD_Z_ann = np.zeros(len(inputs))

JSD_c_beta = np.zeros(len(inputs))
JSD_Z_beta = np.zeros(len(inputs))

bins = np.linspace(0,0.7,20)
bins_center = bins[1:]/2 + bins[:-1]/2
bins_width = bins[1:] - bins[:-1]

for i in range(len(inputs)):
    JSD_c_ann[i] = np.mean(JSDiv(np.atleast_2d(pdf_c_dns[i,:]),np.atleast_2d(pdf_c_ann[i,:])))
    JSD_Z_ann[i] = np.mean(JSDiv(np.atleast_2d(pdf_Z_dns[i,:]),np.atleast_2d(pdf_Z_ann[i,:])))

    JSD_c_beta[i] = np.mean(JSDiv(np.atleast_2d(pdf_c_dns[i,:]),np.atleast_2d(pdf_c_beta[i,:])))
    JSD_Z_beta[i] = np.mean(JSDiv(np.atleast_2d(pdf_Z_dns[i,:]),np.atleast_2d(pdf_Z_beta[i,:])))

h_c, _ = np.histogram(JSD_c_ann,bins)
h_Z, _ = np.histogram(JSD_Z_ann,bins)
pdf_JSD_c_ann = h_c / (np.sum(h_c) * bins_width)
pdf_JSD_Z_ann = h_Z / (np.sum(h_Z) * bins_width)

h_c, _ = np.histogram(JSD_c_beta,bins)
h_Z, _ = np.histogram(JSD_Z_beta,bins)
pdf_JSD_c_beta = h_c / (np.sum(h_c) * bins_width)
pdf_JSD_Z_beta = h_Z / (np.sum(h_Z) * bins_width)

"""
Progress variable
"""
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(bottom=True,top=True,left=True,right=True,direction='in', length=6, width=2, colors='k')
ax.plot(bins_center, pdf_JSD_c_ann, label='ANN', color='blue', linewidth=3, mfc='none')
ax.plot(bins_center, pdf_JSD_c_beta, label=r'$\beta$-PDF', color='orange', linewidth=3, mfc='none')
ax.set_xlabel('JSD', fontsize=18)
ax.set_ylabel('PDF',fontsize=18)
ax.set_xlim([0,0.3])
ax.set_ylim([0,20])
ax.set_xticks([0,0.1,0.2,0.3])
ax.set_xticklabels([0,0.1,0.2,0.3],fontsize=16)
plt.yticks(fontsize=16)
eq = '$\overline{\\text{JSD}}^{\\text{ANN}}$: %.4f\n$\overline{\\text{JSD}}^{\\beta-\\text{PDF}}$: %.4f' %(
    np.mean(JSD_c_ann),np.mean(JSD_c_beta))
ax.text(0.1,14,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)

"""
Mixture fraction
"""
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(bottom=True,top=True,left=True,right=True,direction='in', length=6, width=2, colors='k')
ax.plot(bins_center, pdf_JSD_Z_ann, label='ANN', color='blue', linewidth=3, mfc='none')
ax.plot(bins_center, pdf_JSD_Z_beta, label=r'$\beta$-PDF', color='orange', linewidth=3, mfc='none')
ax.set_xlabel('JSD',fontsize=18)
ax.set_ylabel('PDF',fontsize=18)
ax.set_xlim([0,0.3])
ax.set_ylim([0,20])
ax.set_xticks([0,0.1,0.2,0.3])
ax.set_xticklabels([0,0.1,0.2,0.3],fontsize=16)
plt.yticks(fontsize=16)
eq = '$\overline{\\text{JSD}}^{\\text{ANN}}$: %.4f \n$\overline{\\text{JSD}}^{\\beta-\\text{PDF}}$: %.4f' %(
                                                                        np.mean(JSD_Z_ann),np.mean(JSD_Z_beta))
ax.text(0.1,14,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)

#---- Scatter plot of reaction rate ------------------------------------------------------
int = np.sum(W_avg[None,:,:]*pdf_Z_dns[:,None,:]*dataSet.ZWidth_hist[None,None,:], axis=-1)
W_dns = rhoBar*np.sum(int*pdf_c_dns*dataSet.cWidth_hist[None,:], axis=-1)
W_dns[W_dns<0] = 0

int = np.sum(W_avg[None,:,:]*pdf_Z_ann[:,None,:]*dataSet.ZWidth_hist[None,None,:], axis=-1)
W_ann = rhoBar*np.sum(int*pdf_c_ann*dataSet.cWidth_hist[None,:], axis=-1)
W_ann[W_ann<0] = 0

int = np.sum(W_avg[None,:,:]*pdf_Z_beta[:,None,:]*dataSet.ZWidth_hist[None,None,:], axis=-1)
W_beta = rhoBar*np.sum(int*pdf_c_beta*dataSet.cWidth_hist[None,:], axis=-1)
W_beta[W_beta<0] = 0

RMSE_ann = 0; RMSE_beta = 0
count = 0
for i in range(len(inputs)):
    if W_dns[i] > 1:
        count += 1
        RMSE_ann += np.square((W_ann[i] - W_dns[i]) / W_dns[i])
        RMSE_beta += np.square((W_beta[i] - W_dns[i]) / W_dns[i])
RMSE_ann = np.sqrt(RMSE_ann / count)
RMSE_beta = np.sqrt(RMSE_beta / count)

"""
Against reaction rate calculated by PDF from DNS
"""
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(bottom=True,top=True,left=True,right=True,direction='in', length=6, width=2, colors='k')
ax.scatter(W_dns, W_ann, marker='^', s=100, facecolors='none', edgecolors='blue', label='ANN', linewidth=0.7)
ax.scatter(W_dns, W_beta, marker='s', s=100, facecolors='none', edgecolors='orange', label=r'$\beta$-PDF', linewidth=0.7)
ax.set_xlabel(r'$\overline{\dot{\omega}}_{c}^{\text{m-DNS}}$',fontsize=18)
ax.set_ylabel(r'$\overline{\dot{\omega}}_{c}^{\text{m}}$',fontsize=18)
ax.set_xlim([0,6000])
ax.set_ylim([0,6000])
ax.set_xticks([0,2000,4000,6000])
ax.set_xticklabels([0,2000,4000,6000],fontsize=16)
ax.set_yticks([2000,4000,6000])
ax.set_yticklabels([2000,4000,6000],fontsize=16)
lims = [0, 6000]
_ = ax.plot(lims, lims, 'k--', linewidth=4)
eq = '$\\text{RMSE}_{\\text{ANN}}$: %.4f \n$\\text{RMSE}_{\\beta}$: %.4f' %(
                                            RMSE_ann,RMSE_beta)
ax.text(200,4200,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)

"""
Against reaction rate filtered from DNS
"""
RMSE_bar_ann = 0; RMSE_bar_beta = 0
count = 0
for i in range(len(inputs)):
    if wBar[i] > 1:
        count += 1
        RMSE_bar_ann += np.square((W_ann[i] - wBar[i]) / wBar[i])
        RMSE_bar_beta += np.square((W_beta[i] - wBar[i]) / wBar[i])
RMSE_bar_ann = np.sqrt(RMSE_bar_ann / count)
RMSE_bar_beta = np.sqrt(RMSE_bar_beta / count)

fig = plt.figure(figsize=(7,6))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(bottom=True,top=True,left=True,right=True,direction='in', length=6, width=2, colors='k')
ax.scatter(wBar, W_ann, marker='^', s=100, facecolors='none', edgecolors='blue', label='ANN', linewidth=0.7)
ax.scatter(wBar, W_beta, marker='s', s=100, facecolors='none', edgecolors='orange', label=r'$\beta$-PDF', linewidth=0.7)
ax.set_xlabel(r'$\overline{\dot{\omega}}_{c}^{\text{DNS}}$',fontsize=18)
ax.set_ylabel(r'$\overline{\dot{\omega}}_{c}^{\text{m}}$',fontsize=18)
ax.set_xlim([0,6000])
ax.set_ylim([0,6000])
ax.set_xticks([0,2000,4000,6000])
ax.set_xticklabels([0,2000,4000,6000],fontsize=16)
ax.set_yticks([2000,4000,6000])
ax.set_yticklabels([2000,4000,6000],fontsize=16)
lims = [0, 6000]
_ = ax.plot(lims, lims, 'k--', linewidth=4)
eq = '$\\text{RMSE}_{\\text{ANN}}$: %.4f \n$\\text{RMSE}_{\\beta}$: %.4f' %(
                                            RMSE_bar_ann,RMSE_bar_beta)
ax.text(2000,5200,eq,fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
plt.close(fig)