import os
os.chdir("..")

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import matplotlib
# from src.utils.plotting import make_matching_figure, error_colormap


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=300, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)
    
    for idx0,kpt0 in enumerate (mkpts0):
            axes[0].annotate(str(idx0), (kpt0[0], kpt0[1]), 
            # arrowprops = {
            #     'headwidth': 10, # 箭头头部的宽度
            #     'headlength': 5, # 箭头头部的长度
            #     'width': 4, # 箭头尾部的宽度
            #     'facecolor': 'r', # 箭头的颜色
            #     'shrink': 0.1, # 从箭尾到标注文本内容开始两端空隙长度
            #  },
            #  family='Times New Roman',  # 标注文本字体为Times New Roman
             fontsize=3)
    
    for idx1,kpt1 in enumerate (mkpts1):
            axes[1].annotate(str(idx1), (kpt1[0], kpt1[1]), 
            # arrowprops = {
            #     'headwidth': 10, # 箭头头部的宽度
            #     'headlength': 5, # 箭头头部的长度
            #     'width': 4, # 箭头尾部的宽度
            #     'facecolor': 'r', # 箭头的颜色
            #     'shrink': 0.1, # 从箭尾到标注文本内容开始两端空隙长度
            #  },
            #  family='Times New Roman',  # 标注文本字体为Times New Roman
             fontsize=3)
    

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=10, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

def make_prediction_and_evaluation_plot(root_dir, pe,save_pair_name, path=None, source='ScanNet'):
    img0 = cv2.imread(str(root_dir / pe['pair_names'][0]), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(root_dir / pe['pair_names'][1]), cv2.IMREAD_GRAYSCALE)
    if source == 'ScanNet':
        img0 = cv2.resize(img0, (640, 480))
        img1 = cv2.resize(img1, (640, 480))

    thr = 5e-4
    mkpts0 = pe['mkpts0_f']
    mkpts1 = pe['mkpts1_f']
    color = error_colormap(pe['epi_errs'], thr, alpha=0.3)

    text = [
        f"LoFTR",
        save_pair_name,
        f"#Matches: {len(mkpts0)}",
        f"$\\Delta$R:{pe['R_errs']:.2f}°,  $\\Delta$t:{pe['t_errs']:.2f}°",
    ]
    if path:
        make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=text, path=path)
    else:
        return make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=text)

root_dir = Path("/home/yan/code/LoFTR/data/megadepth/test")  # MegaDepth
save_results_folder = '/home/yan/code/LoFTR/viz_results/loftr_pred_thr0.0010_idx'
npy_path = "/home/yan/code/LoFTR/dump/CCTV7_test_5/loftr_pred_thr0.0010/LoFTR_pred_eval.npy"
dumps = np.load(npy_path, allow_pickle=True)

R_err =[]
t_err =[]
inliers_num =[]
mean_epi_errs = []

for pred in dumps:
    save_pair_name= pred['pair_names'][0].split('/')[-1].split('.')[0]+'-'+ pred['pair_names'][1].split('/')[-1].split('.')[0]+'.JPG'
    save_fig_path = os.path.join(save_results_folder, save_pair_name)
    print(save_fig_path)
    make_prediction_and_evaluation_plot(root_dir,pred,save_pair_name,path=save_fig_path ,source='MegaDepth')
    if pred['R_errs']!= float('inf'):
        R_err.append(pred['R_errs'])
    else:
        print(pred['R_errs'])
    if pred['R_errs']!= float('inf'):
        t_err.append(pred['t_errs'])    
    inliers_num.append(np.sum(pred['inliers']!=0))
    mean_epi_errs.append(np.mean(pred['epi_errs']))

mean_R_err = np.mean(np.array(R_err))
mean_t_err =np.mean(np.array(t_err))
mean_inliers_num =np.mean(np.array(inliers_num))
all_test_mean_epi_errs =  np.mean(np.array(mean_epi_errs))

save_txt_path = os.path.join('/media/yan/data1/lyx/CCTV7', 'metrics.txt')
print(save_txt_path)
print('mean_R_err : '+str(mean_R_err))
print('mean_t_err : '+str(mean_t_err))
print('mean_inliers_num : '+str(mean_inliers_num))
print('all_test_mean_epi_errs : '+str(all_test_mean_epi_errs))


with open(save_fig_path,'w+') as f:
    f.write('mean_R_err : '+str(mean_R_err)+'\n')
    f.write('mean_t_err : '+str(mean_t_err)+'\n')
    f.write('mean_inliers_num : '+str(mean_inliers_num)+'\n')
    f.write('all_test_mean_epi_errs : '+str(all_test_mean_epi_errs))
f.close()



