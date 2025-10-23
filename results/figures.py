# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from kelvin_model import (kelvin_lib_path, prepare_data, calc_density,
                          compute_stress)

if __name__ == '__main__':

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    img = '7LX1'

    ref_img_pth = {'7LX1': Path(f"./data/{img}/hdr/14_5391.npy"),
                   '7LX1_2': Path(f"./data/{img}/hdr/4_2414.npy"),
                   '7LX1_3': Path(f"./data/{img}/hdr/7_04362.npy")}
    ref_img = np.load(ref_img_pth[img])

    lib_path = Path(kelvin_lib_path)

    density_base = np.load(Path(f"./data/{img}/density.npy"))

    if img == '7LX1':
        roi_y = slice(1486, 3005, 1)
        roi_x = slice(1360, 2416, 1)
    elif img == '7LX1_2':
        roi_y = slice(1480, 2897, 1)
        roi_x = slice(698, 1898, 1)
    elif img == '7LX1_3':
        roi_y = slice(1529, 3042, 1)
        roi_x = slice(1229, 2190, 1)
    else:
        raise ValueError
    density_base = density_base[roi_x, roi_y]

    gauss_fit = np.load(Path(f"./data/{img}/fit.npy"))

    peaks = np.radians(np.load(Path(f"./data/{img}/angle.npy")))

    # Images to use for the optimization
    def_images_paths = {
        '7LX1': (Path(f"./data/{img}/hdr/15_5570.npy"),
                 Path(f"./data/{img}/hdr/21_6630.npy")),
        '7LX1_2': (Path(f"./data/{img}/hdr/5_2592.npy"),
                   Path(f"./data/{img}/hdr/13_4018.npy")),
        '7LX1_3': (Path(f"./data/{img}/hdr/8_04540.npy"),
                   Path(f"./data/{img}/hdr/16_05955.npy"))}

    def_images = tuple(np.load(image) for image in def_images_paths[img])

    nb_interp_diag = 200
    diagonal_downscaling = 20

    # Other parameters driving the optimization process
    fit_file = pd.read_csv(Path(f"./data/{img}/results.csv"))
    results_file = pd.read_csv(Path(f"./data/{img}/comparison.csv"))

    (exxs, eyys, exys,
     sigma_1, sigma_2, sigma_3,
     theta_1, theta_2, theta_3,
     m_1, m_2, m_3,
     interp_pts, normals) = prepare_data(ref_img,
                                         gauss_fit,
                                         peaks,
                                         def_images,
                                         nb_interp_diag,
                                         diagonal_downscaling)

    density = calc_density(density_base, fit_file['density_min'].iloc[0],
                           fit_file['contrast'].iloc[0])

    fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10, 15))

    axs[0][0].imshow(ref_img, cmap='Greys', clim=(0, 1))
    axs[0][0].set_title('(a)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Greys'),
                     ax=axs[0][0])
    c.set_label('Pixel intensity', fontsize=12)

    axs[0][1].imshow(density, cmap='magma', clim=(0, 1))
    axs[0][1].set_title('(b)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'),
                     ax=axs[0][1])
    c.set_label(r'$\rho$', fontsize=12)

    hi, lo = np.percentile(sigma_1, 99), np.percentile(sigma_1, 1)
    axs[1][0].imshow(np.clip(sigma_1, lo, hi), cmap='magma',
                     clim=(0, hi))
    axs[1][0].set_title('(c)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(0, hi), cmap='magma'),
                     ax=axs[1][0])
    c.set_label(r'$\sigma_1$', fontsize=12)

    axs[1][1].imshow(np.rad2deg(theta_1), cmap='twilight', clim=(0, 180))
    axs[1][1].set_title('(d)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                     ax=axs[1][1])
    c.set_label(r'$\mu_1$', fontsize=12)

    lo = -max(abs(np.percentile(exxs[0], 1)), abs(np.percentile(exxs[0], 99)))
    hi = max(abs(np.percentile(exxs[0], 1)), abs(np.percentile(exxs[0], 99)))
    axs[2][0].imshow(exxs[0], cmap='coolwarm', clim=(lo, hi))
    axs[2][0].set_title('(e)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=axs[2][0])
    c.set_label(r'Low $H_{xx}$', fontsize=12)

    lo = -max(abs(np.percentile(exxs[1], 1)), abs(np.percentile(exxs[1], 99)))
    hi = max(abs(np.percentile(exxs[1], 1)), abs(np.percentile(exxs[1], 99)))
    axs[2][1].imshow(exxs[1], cmap='coolwarm', clim=(lo, hi))
    axs[2][1].set_title('(f)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=axs[2][1])
    c.set_label(r'Low $H_{yy}$', fontsize=12)

    lo = -max(abs(np.percentile(eyys[0], 1)), abs(np.percentile(eyys[0], 99)))
    hi = max(abs(np.percentile(eyys[0], 1)), abs(np.percentile(eyys[0], 99)))
    axs[3][0].imshow(eyys[0], cmap='coolwarm', clim=(lo, hi))
    axs[3][0].set_title('(g)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=axs[3][0])
    c.set_label(r'High $H_{xx}$', fontsize=12)

    lo = -max(abs(np.percentile(eyys[1], 1)), abs(np.percentile(eyys[1], 99)))
    hi = max(abs(np.percentile(eyys[1], 1)), abs(np.percentile(eyys[1], 99)))
    axs[3][1].imshow(eyys[1], cmap='coolwarm', clim=(lo, hi))
    axs[3][1].set_title('(h)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=axs[3][1])
    c.set_label(r'High $H_{yy}$', fontsize=12)

    axs[4][0].imshow(m_1, cmap='magma', clim=(-1, 1))
    axs[4][0].set_title('(i)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap='magma'),
                     ax=axs[4][0])
    c.set_label(r'$m_1$', fontsize=12)

    fig.delaxes(axs[4][1])

    plt.savefig(f'./input_{img}.svg', dpi=300)

    sxx_lo, syy_lo, sxy_lo, _ = compute_stress(lib_path,
                                               exxs[0],
                                               eyys[0],
                                               exys[0],
                                               m_1,
                                               m_2,
                                               m_3,
                                               fit_file['lambda_h'].iloc[0],
                                               fit_file['lambda_11'].iloc[0],
                                               fit_file['lambda_21'].iloc[0],
                                               fit_file['lambda_51'].iloc[0],
                                               fit_file['lambda_12'].iloc[0],
                                               fit_file['lambda_22'].iloc[0],
                                               fit_file['lambda_52'].iloc[0],
                                               fit_file['lambda_13'].iloc[0],
                                               fit_file['lambda_23'].iloc[0],
                                               fit_file['lambda_53'].iloc[0],
                                               fit_file['lambda_14'].iloc[0],
                                               fit_file['lambda_24'].iloc[0],
                                               fit_file['lambda_54'].iloc[0],
                                               fit_file['lambda_15'].iloc[0],
                                               fit_file['lambda_25'].iloc[0],
                                               fit_file['lambda_55'].iloc[0],
                                               fit_file['val1'].iloc[0],
                                               fit_file['val2'].iloc[0],
                                               fit_file['val3'].iloc[0],
                                               fit_file['val4'].iloc[0],
                                               fit_file['val5'].iloc[0],
                                               theta_1,
                                               theta_2,
                                               theta_3,
                                               sigma_1,
                                               sigma_2,
                                               sigma_3,
                                               density)

    sxx_hi, syy_hi, sxy_hi, _ = compute_stress(lib_path,
                                               exxs[1],
                                               eyys[1],
                                               exys[1],
                                               m_1,
                                               m_2,
                                               m_3,
                                               fit_file['lambda_h'].iloc[0],
                                               fit_file['lambda_11'].iloc[0],
                                               fit_file['lambda_21'].iloc[0],
                                               fit_file['lambda_51'].iloc[0],
                                               fit_file['lambda_12'].iloc[0],
                                               fit_file['lambda_22'].iloc[0],
                                               fit_file['lambda_52'].iloc[0],
                                               fit_file['lambda_13'].iloc[0],
                                               fit_file['lambda_23'].iloc[0],
                                               fit_file['lambda_53'].iloc[0],
                                               fit_file['lambda_14'].iloc[0],
                                               fit_file['lambda_24'].iloc[0],
                                               fit_file['lambda_54'].iloc[0],
                                               fit_file['lambda_15'].iloc[0],
                                               fit_file['lambda_25'].iloc[0],
                                               fit_file['lambda_55'].iloc[0],
                                               fit_file['val1'].iloc[0],
                                               fit_file['val2'].iloc[0],
                                               fit_file['val3'].iloc[0],
                                               fit_file['val4'].iloc[0],
                                               fit_file['val5'].iloc[0],
                                               theta_1,
                                               theta_2,
                                               theta_3,
                                               sigma_1,
                                               sigma_2,
                                               sigma_3,
                                               density)

    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    lo = -max(abs(np.percentile(sxx_lo, 1)), abs(np.percentile(sxx_lo, 99)))
    hi = max(abs(np.percentile(sxx_lo, 1)), abs(np.percentile(sxx_lo, 99)))
    ax1.imshow(sxx_lo, cmap='coolwarm', clim=(lo, hi))
    ax1.set_title('(a)', loc='left')
    ax1.text(0.4, 1.07, "Low strain", size=12, verticalalignment='center',
             transform=ax1.transAxes)
    plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                 ax=ax1)
    ax1.set_xticklabels([])

    lo = -max(abs(np.percentile(sxx_hi, 1)), abs(np.percentile(sxx_hi, 99)))
    hi = max(abs(np.percentile(sxx_hi, 1)), abs(np.percentile(sxx_hi, 99)))
    ax2.imshow(sxx_hi, cmap='coolwarm', clim=(lo, hi))
    ax2.set_title('(b)', loc='left')
    ax2.text(0.4, 1.07, "High strain", size=12, verticalalignment='center',
             transform=ax2.transAxes)
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=ax2)
    c.set_label(r'$\tau_{xx}$', fontsize=12)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    lo = -max(abs(np.percentile(syy_lo, 1)), abs(np.percentile(syy_lo, 99)))
    hi = max(abs(np.percentile(syy_lo, 1)), abs(np.percentile(syy_lo, 99)))
    ax3.imshow(syy_lo, cmap='coolwarm', clim=(lo, hi))
    ax3.set_title('(c)', loc='left')
    plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                 ax=ax3)

    lo = -max(abs(np.percentile(syy_hi, 1)), abs(np.percentile(syy_hi, 99)))
    hi = max(abs(np.percentile(syy_hi, 1)), abs(np.percentile(syy_hi, 99)))
    ax4.imshow(syy_hi, cmap='coolwarm', clim=(lo, hi))
    ax4.set_title('(d)', loc='left')
    c = plt.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='coolwarm'),
                     ax=ax4)
    c.set_label(r'$\tau_{yy}$', fontsize=12)
    ax4.set_yticklabels([])

    ax5.plot(results_file['measured_pos'].values / ref_img.shape[1] * 100,
             results_file['measured_x'].values, marker='+', color='k',
             label=r'Expe. data ($x$)', linestyle='none')
    ax5.plot(results_file['measured_pos'].values / ref_img.shape[1] * 100,
             results_file['calculated_x'].values, color='royalblue',
             label=r'Comp. force ($x$)')
    ax5.plot(results_file['measured_pos'].values / ref_img.shape[1] * 100,
             results_file['calculated_y'].values, color='sandybrown',
             label=r'Comp. force ($y$)')
    ax5.set_xlabel('Macroscopic strain (mm/mm)')
    ax5.set_ylabel('Effort (N)')
    ax5.set_title('(e)', loc='left')
    ax5.legend()

    plt.savefig(f'./output_{img}.svg', dpi=300)
