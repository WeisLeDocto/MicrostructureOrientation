# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from kelvin_model import (kelvin_lib_path, prepare_data, calc_density,
                          compute_stress, stress_diag_to_force,
                          diagonals_interpolator)

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
        '7LX1': (Path(f"./data/{img}/hdr/21_6630.npy"),),
        '7LX1_2': (Path(f"./data/{img}/hdr/14_4196.npy"),),
        '7LX1_3': (Path(f"./data/{img}/hdr/16_05955.npy"),)}

    def_images = tuple(np.load(image) for image in def_images_paths[img])

    nb_interp_diag = 200
    diagonal_downscaling = 20
    scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    fit_file = pd.read_csv(Path(f"./data/{img}/results.csv"))

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

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x_ref, force_y_ref = stress_diag_to_force(sxx,
                                                    syy,
                                                    sxy,
                                                    interp_pts,
                                                    normals,
                                                    scale,
                                                    thickness)

    results = dict()

    sig_1_avg = np.full_like(sigma_1, np.average(sigma_1))
    sig_2_avg = np.full_like(sigma_2,
                             np.average(sigma_2[~np.isnan(peaks[..., 1])]))
    sig_2_avg[np.isnan(peaks[..., 1])] = 0.0
    sig_3_avg = np.full_like(sigma_3,
                             np.average(sigma_3[~np.isnan(peaks[..., 2])]))
    sig_3_avg[np.isnan(peaks[..., 2])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sig_1_avg,
                                                            sig_2_avg,
                                                            sig_3_avg,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['sigma'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                           np.median(force_x_ref))

    dens_avg = np.full_like(density, np.average(density))

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            dens_avg)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['density'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                             np.median(force_x_ref))

    avg = (0.5 * np.atan2(np.sum(np.sin(2 * theta_1)),
                          np.sum(np.cos(2 * theta_1)))) % np.pi
    ang_1_avg = np.full_like(theta_1, avg)
    avg = ((0.5 * np.atan2(np.sum(np.sin(2 *
                                         theta_2[~np.isnan(peaks[..., 1])])),
                           np.sum(np.cos(2 *
                                         theta_2[~np.isnan(peaks[..., 1])]))))
           % np.pi)
    ang_2_avg = np.full_like(theta_2, avg)
    ang_2_avg[np.isnan(peaks[..., 1])] = 0.0
    avg = ((0.5 * np.atan2(np.sum(np.sin(2 *
                                         theta_3[~np.isnan(peaks[..., 2])])),
                           np.sum(np.cos(2 *
                                         theta_3[~np.isnan(peaks[..., 2])]))))
           % np.pi)
    ang_3_avg = np.full_like(theta_3, avg)
    ang_3_avg[np.isnan(peaks[..., 2])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            ang_1_avg,
                                                            ang_2_avg,
                                                            ang_3_avg,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['angle'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                           np.median(force_x_ref))

    m_1_avg = np.full_like(m_1, np.average(m_1))
    m_2_avg = np.full_like(m_2, np.average(m_2[~np.isnan(peaks[..., 1])]))
    m_2_avg[np.isnan(peaks[..., 1])] = 0.0
    m_3_avg = np.full_like(m_3, np.average(m_3[~np.isnan(peaks[..., 2])]))
    m_3_avg[np.isnan(peaks[..., 1])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1_avg,
                                                            m_2_avg,
                                                            m_3_avg,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['m'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                       np.median(force_x_ref))

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1_avg,
                                                            m_2_avg,
                                                            m_3_avg,
                                                            interp_pts,
                                                            ang_1_avg,
                                                            ang_2_avg,
                                                            ang_3_avg,
                                                            sig_1_avg,
                                                            sig_2_avg,
                                                            sig_3_avg,
                                                            dens_avg)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['all'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                         np.median(force_x_ref))

    fig = plt.figure(figsize=(17, 3))
    ax = fig.add_subplot(1, 3, 1)

    ax.vlines(0, -0.225, 2.225, color='k')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.yaxis.set_tick_params(length=0, labelsize=12)
    ax.set_title(img, fontsize=12)

    ax.barh((1.5, 1.0, 0.5, 0.0, 2.0),
            results.values(), align='center', height=0.45,
            tick_label=results.keys())

    ax.set_xlim(0, ax.get_xlim()[1])

    img = '7LX1_2'

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
        '7LX1': (Path(f"./data/{img}/hdr/21_6630.npy"),),
        '7LX1_2': (Path(f"./data/{img}/hdr/14_4196.npy"),),
        '7LX1_3': (Path(f"./data/{img}/hdr/18_06311.npy"),)}

    def_images = tuple(np.load(image) for image in def_images_paths[img])

    nb_interp_diag = 200
    diagonal_downscaling = 20
    scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    fit_file = pd.read_csv(Path(f"./data/{img}/results.csv"))

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

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x_ref, force_y_ref = stress_diag_to_force(sxx,
                                                    syy,
                                                    sxy,
                                                    interp_pts,
                                                    normals,
                                                    scale,
                                                    thickness)

    results = dict()

    sig_1_avg = np.full_like(sigma_1, np.average(sigma_1))
    sig_2_avg = np.full_like(sigma_2,
                             np.average(sigma_2[~np.isnan(peaks[..., 1])]))
    sig_2_avg[np.isnan(peaks[..., 1])] = 0.0
    sig_3_avg = np.full_like(sigma_3,
                             np.average(sigma_3[~np.isnan(peaks[..., 2])]))
    sig_3_avg[np.isnan(peaks[..., 2])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sig_1_avg,
                                                            sig_2_avg,
                                                            sig_3_avg,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['sigma'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                           np.median(force_x_ref))

    dens_avg = np.full_like(density, np.average(density))

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            dens_avg)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['density'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                             np.median(force_x_ref))

    avg = (0.5 * np.atan2(np.sum(np.sin(2 * theta_1)),
                          np.sum(np.cos(2 * theta_1)))) % np.pi
    ang_1_avg = np.full_like(theta_1, avg)
    avg = ((0.5 * np.atan2(np.sum(np.sin(2 *
                                         theta_2[~np.isnan(peaks[..., 1])])),
                           np.sum(np.cos(2 *
                                         theta_2[~np.isnan(peaks[..., 1])]))))
           % np.pi)
    ang_2_avg = np.full_like(theta_2, avg)
    ang_2_avg[np.isnan(peaks[..., 1])] = 0.0
    avg = ((0.5 * np.atan2(np.sum(np.sin(2 *
                                         theta_3[~np.isnan(peaks[..., 2])])),
                           np.sum(np.cos(2 *
                                         theta_3[~np.isnan(peaks[..., 2])]))))
           % np.pi)
    ang_3_avg = np.full_like(theta_3, avg)
    ang_3_avg[np.isnan(peaks[..., 2])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1,
                                                            m_2,
                                                            m_3,
                                                            interp_pts,
                                                            ang_1_avg,
                                                            ang_2_avg,
                                                            ang_3_avg,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['angle'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                           np.median(force_x_ref))

    m_1_avg = np.full_like(m_1, np.average(m_1))
    m_2_avg = np.full_like(m_2, np.average(m_2[~np.isnan(peaks[..., 1])]))
    m_2_avg[np.isnan(peaks[..., 1])] = 0.0
    m_3_avg = np.full_like(m_3, np.average(m_3[~np.isnan(peaks[..., 2])]))
    m_3_avg[np.isnan(peaks[..., 1])] = 0.0

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1_avg,
                                                            m_2_avg,
                                                            m_3_avg,
                                                            interp_pts,
                                                            theta_1,
                                                            theta_2,
                                                            theta_3,
                                                            sigma_1,
                                                            sigma_2,
                                                            sigma_3,
                                                            density)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['m'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                       np.median(force_x_ref))

    (exx_diags, eyy_diags, exy_diags, m_1_diags, m_2_diags, m_3_diags,
     theta_1_diags, theta_2_diags, theta_3_diags, sigma_1_diags, sigma_2_diags,
     sigma_3_diags, density_diags) = diagonals_interpolator(exxs[0],
                                                            eyys[0],
                                                            exys[0],
                                                            m_1_avg,
                                                            m_2_avg,
                                                            m_3_avg,
                                                            interp_pts,
                                                            ang_1_avg,
                                                            ang_2_avg,
                                                            ang_3_avg,
                                                            sig_1_avg,
                                                            sig_2_avg,
                                                            sig_3_avg,
                                                            dens_avg)

    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   m_1_diags,
                                   m_2_diags,
                                   m_3_diags,
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
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)

    force_x, force_y = stress_diag_to_force(sxx,
                                            syy,
                                            sxy,
                                            interp_pts,
                                            normals,
                                            scale,
                                            thickness)

    results['all'] = abs((np.median(force_x) - np.median(force_x_ref)) /
                         np.median(force_x_ref))

    ax = fig.add_subplot(1, 3, 2)

    ax.vlines(0, -0.225, 2.225, color='k')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.yaxis.set_tick_params(length=0, labelsize=12)
    ax.set_title(img, fontsize=12)

    ax.barh((0.0, 2.0, 1.5, 0.5, 1.0),
            results.values(), align='center', height=0.45,
            tick_label=results.keys())

    ax.set_xlim(0, ax.get_xlim()[1])

    plt.show()
