/*!
 * \file   KelvinDecomposition.hxx
 * \brief    
 * \date   25/10/2024
 */

#ifndef LIB_KELVINDECOMPOSITION_HXX
#define LIB_KELVINDECOMPOSITION_HXX

#include "TFEL/Math/st2tost2.hxx"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace tfel::math{
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    kelvin_integrated_tensor(NumericType lambda_h, 
                             NumericType lambda_1, 
                             NumericType lambda_2, 
                             NumericType lambda_3, 
                             NumericType lambda_4, 
                             NumericType lambda_5, 
                             NumericType sigma,
                             NumericType m,
                             bool skip_z = false) {
    
    st2tost2<3u, NumericType> integrated;
    
    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        integrated(i, j) = 0.0;
      }
    }
    
    const double lamh_m = pow(lambda_h, m);
    const double lam1_m = pow(lambda_1, m);
    const double lam2_m = pow(lambda_2, m);
    const double lam3_m = pow(lambda_3, m);
    const double lam4_m = pow(lambda_4, m);
    const double lam5_m = pow(lambda_5, m);
    
    const double sigma_sq = sigma * sigma;
    
    const double exp_2s = exp(2.0 * sigma_sq);
    const double exp_6s = exp_2s * exp_2s * exp_2s;
    const double exp_8s = exp_2s * exp_6s;
    const double exp_neg2s = 1.0 / exp_2s;
    const double exp_neg8s = 1.0 / exp_8s;
    
    const double term_A = 11.0 * lam1_m + 9.0 * lam2_m + 12.0 * lam5_m 
        + 16.0 * lamh_m;
    const double term_B = 12.0 * (lam1_m - lam2_m);
    const double term_C = 9.0 * lam1_m + 3.0 * lam2_m - 12.0 * lam5_m;
    const double term_D = 7.0 * lam1_m - 3.0 * lam2_m + 12.0 * lam5_m 
        - 16.0 * lamh_m;
    const double term_E = lam1_m + 3.0 * lam2_m - 4.0 * lamh_m;
    const double term_F = 3.0 * lam1_m + lam2_m + 4.0 * lam5_m;
    const double term_G = 3.0 * lam1_m - 3.0 * lam2_m;
    
    integrated(0, 0) = 1.0 / 48.0 * (term_A + term_B * exp_neg2s + term_C 
        * exp_neg8s);
    integrated(0, 1) = -1.0 / 48.0 * (term_D + term_C * exp_neg8s);
    integrated(0, 2) = -1.0 / 12.0 * (term_E + term_G * exp_neg2s);
    integrated(1, 1) = 1.0 / 48.0 * (term_A - term_B * exp_neg2s + term_C 
        * exp_neg8s);
    integrated(1, 2) = -1.0 / 12.0 * (term_E - term_G * exp_neg2s);
    integrated(2, 2) = 1.0 / 6.0 * lam1_m + 1.0 / 2.0 * lam2_m + 1.0 / 3.0 
        * lamh_m;
    integrated(3, 3) = 1.0 / 8.0 * (term_F + (-3.0 * lam1_m - lam2_m + 4.0 
        * lam5_m) * exp_neg8s);
    
    if (skip_z) {
      integrated(4, 4) = 1.0;
      integrated(5, 5) = 1.0;
    }
    else {
      const double term_H = lam3_m + lam4_m;
      const double term_J = lam3_m - lam4_m;
      integrated(4, 4) = 0.5 * (term_H - term_J * exp_neg2s);
      integrated(5, 5) = 0.5 * (term_H + term_J * exp_neg2s);
    }
      
    for (int i = 0; i < 6; i++){
      for (int j = 0; j < i; j++){
        integrated(i, j) = integrated(j, i);
      }
    }
    
    return integrated;
    
  }
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    zero_m_approximation(const NumericType lambda_h, 
                         const NumericType lambda_1, 
                         const NumericType lambda_2, 
                         const NumericType lambda_3, 
                         const NumericType lambda_4, 
                         const NumericType lambda_5, 
                         const NumericType sigma,
                         const NumericType m,
                         bool skip_z = false) {
    
    st2tost2<3u, NumericType> integrated;

    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        integrated(i, j) = 0.0;
      }
    }
    
    const double lamh_l = log(lambda_h);
    const double lam1_l = log(lambda_1);
    const double lam2_l = log(lambda_2);
    const double lam3_l = log(lambda_3);
    const double lam4_l = log(lambda_4);
    const double lam5_l = log(lambda_5);
    
    const double sigma_sq = sigma * sigma;
    
    const double exp_2s = exp(2.0 * sigma_sq);
    const double exp_4s = exp_2s * exp_2s;
    const double exp_6s = exp_2s * exp_4s;
    const double exp_8s = exp_4s * exp_4s;
    const double exp_10s = exp_8s * exp_2s;
    const double exp_12s = exp_8s * exp_4s;
    const double exp_14s = exp_12s * exp_2s;
    const double exp_16s = exp_8s * exp_8s;
    
    const double exp_neg2s = 1.0 / exp_2s;
    const double exp_neg4s = 1.0 / exp_4s;
    const double exp_neg8s = 1.0 / exp_8s;
    const double exp_neg10s = 1.0 / exp_10s;
    const double exp_neg16s = 1.0 / exp_16s;
    
    const double term_A = 11.0 * lam1_l + 9.0 * lam2_l + 12.0 * lam5_l + 16.0 
        * lamh_l;
    const double term_B = 12.0 * (lam1_l - lam2_l);
    const double term_C = 9.0 * lam1_l + 3.0 * lam2_l - 12.0 * lam5_l;
    const double term_D = 7.0 * lam1_l - 3.0 * lam2_l + 12.0 * lam5_l - 16.0 
        * lamh_l;
    const double term_E = lam1_l + 3.0 * lam2_l - 4.0 * lamh_l;
    const double term_F = 3.0 * lam1_l + lam2_l + 4.0 * lam5_l;
    const double term_G = 3.0 * lam1_l - 3.0 * lam2_l;
    
    const double lam1_sq = lam1_l * lam1_l;
    const double lam2_sq = lam2_l * lam2_l;
    const double lam5_sq = lam5_l * lam5_l;
    const double lam1_lam2 = lam1_l * lam2_l;
    const double lam1_lam5 = lam1_l * lam5_l;
    const double lam2_lam5 = lam2_l * lam5_l;
    const double cross_term = lam1_sq - 2.0 * lam1_lam2 + lam2_sq;
    
    const double cross_term_m = cross_term * m;
    const double quad_term1 = 19.0 * lam1_sq - 14.0 * lam1_lam2 + 11.0 
        * lam2_sq - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;
    const double quad_term2 = 3.0 * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 
        * lam1_lam5 + 4.0 * lam2_lam5;
    const double quad_term3 = 9.0 * lam1_sq + 6.0 * lam1_lam2 + lam2_sq - 24.0 
        * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;
    const double quad_term4 = 15.0 * lam1_sq - 6.0 * lam1_lam2 + 7.0 * lam2_sq 
        - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;

    integrated(0, 0) = 1.0 / 48.0 * (term_A * exp_8s + term_B * exp_6s 
        + term_C) * exp_neg8s;
    integrated(0, 1) = -1.0 / 48.0 * (term_D * exp_8s + term_C) * exp_neg8s;
    integrated(0, 2) = -1.0 / 12.0 * (term_E * exp_2s + term_G) * exp_neg2s;
    integrated(1, 1) = 1.0 / 48.0 * (term_A * exp_8s - term_B * exp_6s 
        + term_C) * exp_neg8s;
    integrated(1, 2) = -1.0 / 12.0 * (term_E * exp_2s - term_G) * exp_neg2s;
    integrated(2, 2) = 1.0 / 6.0 * lam1_l + 1.0 / 2.0 * lam2_l + 1.0 / 3.0 
        * lamh_l;
    integrated(3, 3) = 1.0 / 8.0 * (term_F * exp_8s - 3.0 * lam1_l - lam2_l 
        + 4.0 * lam5_l) * exp_neg8s;
    if (skip_z) {
    integrated(4, 4) = 1.0;
    integrated(5, 5) = 1.0;
    }
    else {
      const double term_H = lam3_l + lam4_l;
      const double term_J = lam3_l - lam4_l;
      integrated(4, 4) = 0.5 * (term_H * exp_2s - term_J) * exp_neg2s;
      integrated(5, 5) = 0.5 * (term_H * exp_2s + term_J) * exp_neg2s;
    }
    
    integrated(0, 0) += 1.0 / 256.0 * m * (quad_term1 * exp_16s +  4.0 
        * quad_term2 * exp_14s - 16.0 * cross_term * exp_12s + 6.0 * cross_term 
        * exp_8s - 4.0 * quad_term2 * exp_6s - quad_term3) * exp_neg16s;
    integrated(0, 1) += -1.0 / 256.0 * m * ((11.0 * lam1_sq + 2.0 * lam1_lam2 
        + 3.0 * lam2_sq - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq) 
        * exp_16s - 8.0 * cross_term * exp_12s + 6.0 * cross_term * exp_8s 
        - quad_term3) * exp_neg16s;
    integrated(0, 2) += -1.0 / 64.0 * m * (2.0 * cross_term * exp_10s + (3.0 
        * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 
        * lam2_lam5) * exp_8s - 2.0 * cross_term * exp_6s - (3.0 * lam1_sq 
        - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 * lam2_lam5)) 
        * exp_neg10s;
    integrated(1, 1) += 1.0 / 256.0 * m * (quad_term1 * exp_16s - 4.0 
        * quad_term2 * exp_14s - 16.0 * cross_term * exp_12s + 6.0 * cross_term 
        * exp_8s + 4.0 * quad_term2 * exp_6s - quad_term3) * exp_neg16s;
    integrated(1, 2) += -1.0 / 64.0 * m * (2.0 * cross_term * exp_10s - (3.0 
        * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 
        * lam2_lam5) * exp_8s - 2.0 * cross_term * exp_6s + (3.0 * lam1_sq 
        - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 * lam2_lam5)) 
        * exp_neg10s;
    integrated(2, 2) += 1.0 / 16.0 * cross_term_m * (exp_4s - 1.0) * exp_neg4s;
    integrated(3, 3) += 1.0 / 128.0 * m * (quad_term4 * exp_16s - 6.0 
        * cross_term * exp_8s - quad_term3) * exp_neg16s;
    
    if (!skip_z) {
      const double z_cross = lam3_l * lam3_l - 2.0 * lam3_l * lam4_l + lam4_l 
          * lam4_l;
      const double z_term = z_cross * m * (exp_4s - 1.0) * exp_neg4s;
      integrated(4, 4) += 0.125 * z_term;
      integrated(5, 5) += 0.125 * z_term;
    }

    for (int i = 0; i < 6; i++){
      for (int j = 0; j < i; j++){
        integrated(i, j) = integrated(j, i);
      }
    }
    
    return integrated;
    
  }
  
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    pow_1_over_m(const st2tost2<3u, NumericType> integrated_mfront, 
                 const NumericType m) {
    
    using namespace Eigen;
    
    Matrix<NumericType, 6, 6> integrated;

    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        integrated(i, j) = integrated_mfront(i, j);
      }
    }

    /*
    SelfAdjointEigenSolver<Matrix<NumericType, 6, 6>> eigensolver(integrated);
    if (eigensolver.info() != Success) abort();
    
    typename SelfAdjointEigenSolver<Matrix<NumericType, 6, 6>>::RealVectorType 
      eigenvalues = eigensolver.eigenvalues();
    
    typename SelfAdjointEigenSolver<Matrix<NumericType, 
                                    6, 6>>::EigenvectorsType 
      eigenvectors = eigensolver.eigenvectors();
    
    eigenvalues = eigenvalues.array().pow(1 / m);
    
    Matrix<NumericType, 6, 6> homogenized = 
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    */
    
    const Matrix<NumericType, 6, 6> homogenized = integrated.pow(1 / m);
    
    st2tost2<3u, NumericType> homogenized_mfront;

    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        homogenized_mfront(i, j) = homogenized(i, j);
      }
    }
    
    return homogenized_mfront;
    
  }
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    tensor_exponential(const st2tost2<3u, NumericType> integrated_mfront) {
    
    using namespace Eigen;
    
    Matrix<NumericType, 6, 6> integrated;

    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        integrated(i, j) = integrated_mfront(i, j);
      }
    }
    
    const Matrix<NumericType, 6, 6> homogenized = integrated.exp();
    
    st2tost2<3u, NumericType> homogenized_mfront;

    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        homogenized_mfront(i, j) = homogenized(i, j);
      }
    }
    
    return homogenized_mfront;
    
  }
  
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    ellipsoid_stiffness(NumericType lambda_h, NumericType r1, NumericType r2) {
    
    st2tost2<3u, NumericType> stiffness;
    
    for (int i = 0; i < 6; i++){
      for(int j = 0; j < 6; j++){
        stiffness(i, j) = 0.0;
      }
    }
    
    stiffness(0, 0) = 8.0 / 27.0 + 4.0 / 27.0 * r1 + 
      2.0 / 27.0 * pow(r1, 2.0) + 2.0 / 27.0 * (2.0 - r1) * r2 + 
      2.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(0, 1) = -4.0 / 27.0 - 2.0 / 27.0 * r1 - 
      1.0 / 27.0 * pow(r1, 2.0) - 1.0 / 27.0 * (2.0 - r1) * r2 - 
      1.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(0, 2) = -4.0 / 27.0 - 2.0 / 27.0 * r1 - 
      1.0 / 27.0 * pow(r1, 2.0) - 1.0 / 27.0 * (2.0 - r1) * r2 - 
      1.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(1, 1) = 2.0 / 27.0 + 1.0 / 27.0 * r1 + 
      5.0 / 27.0 * pow(r1, 2.0) + 1.0 / 54.0 * (2.0 - r1) * r2 + 
      1.0 / 6.0 * r1 * r2 + 5.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(1, 2) = 2.0 / 27.0 + 1.0 / 27.0 * r1 - 
      4.0 / 27.0 * pow(r1, 2.0) + 1.0 / 54.0 * (2.0 - r1) * r2 - 
      1.0 / 6.0 * r1 * r2 - 4.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(2, 2) = 2.0 / 27.0 + 1.0 / 27.0 * r1 + 
      5.0 / 27.0 * pow(r1, 2.0) + 1.0 / 54.0 * (2.0 - r1) * r2 + 
      1.0 / 6.0 * r1 * r2 + 5.0 / 27.0 * pow(r2, 2.0) + 1.0 / 3.0 * lambda_h;
    stiffness(3, 3) = r1;
    stiffness(4, 4) = r2;
    stiffness(5, 5) = r1 * r2;
    
    for (int i = 0; i < 6; i++){
      for (int j = 0; j < i; j++){
        stiffness(i, j) = stiffness(j, i);
      }
    }
    
    return stiffness;
    
  }
  
  
  template <typename NumericType> st2tost2<3u, NumericType> 
    rotate(const st2tost2<3u, NumericType> homogenized, 
           const NumericType theta) {
    
    tmatrix<3, 3, NumericType> rot_mat;
    rot_mat(0, 0) = cos(-theta);
    rot_mat(0, 1) = -sin(-theta);
    rot_mat(0, 2) = 0.0;
    rot_mat(1, 0) = sin(-theta);
    rot_mat(1, 1) = cos(-theta);
    rot_mat(1, 2) = 0.0;
    rot_mat(2, 0) = 0.0;
    rot_mat(2, 1) = 0.0;
    rot_mat(2, 2) = 1.0;
    
    const st2tost2<3u, NumericType> rot_tens = 
      st2tost2<3u, NumericType>::fromRotationMatrix(rot_mat);
    
    return transpose(rot_tens) * homogenized * rot_tens;
    
  }
  
  
  template <typename NumericType> st2tost2<3u, NumericType> calc_stiffness(
      const std::array<st2tost2<3u, NumericType>, 5> &stiffnesses,
      const stensor<3, NumericType> strain,
      const std::array<NumericType, 5> &vals) {
      
    // Initialize order 0
    st2tost2<3u, NumericType> stiff_tot = vals[0] * stiffnesses[0];

    // Pre-compute the valid orders
    std::vector<size_t> valid_orders;
    for (size_t j = 1; j < vals.size(); ++j) {
      if (vals[j] > 1.0e-12) valid_orders.push_back(j);
    }

    // Iterate over all the valid orders
    for (size_t j : valid_orders) {

      // For each order greater than 1, there is a multiplicative factor
      // dependent on the strain and stiffness
      NumericType factor = strain | (stiffnesses[j] * strain);

      /// The total equivalent stiffness is the sum of the ones for each order
      stiff_tot += vals[j] * (j + 1) * pow(factor, j) * stiffnesses[j];
    }
    
    return stiff_tot;
    
  }
  
  
  template <typename NumericType> NumericType
    calc_ezz_plane_stress(const std::array<st2tost2<3u, NumericType>, 5> 
                              &stiffnesses,
                          stensor<3, NumericType> strain,
                          const std::array<NumericType, 5> &vals,
                          const NumericType stop_crit,
                          const int max_iter) {
    
    st2tost2<3u, NumericType> stiff_tot = calc_stiffness(stiffnesses, 
                                                         strain, vals);
    NumericType szz = (stiff_tot * strain)(2);

    // Stop here if the stress is already close enough to 0
    if (abs(szz) < stop_crit) return strain(2);

    // Use a fixed-point method to compute the optimal strain value
    if (stiff_tot(2, 2) != 0.0) {
      strain(2) = -(stiff_tot(2, 0) * strain(0) + 
                    stiff_tot(2, 1) * strain(1) + 
                    stiff_tot(2, 3) * strain(3)) / stiff_tot(2, 2);
    }
    // In case the stiffness associated with zz is zero, nothing can be done
    else {
      return stiff_tot(2, 0) * strain(0) + stiff_tot(2, 1) * strain(1) 
          + stiff_tot(2, 3) * strain(3);
    }

    // Iterate until reaching a solution or until reaching max iterations
    int n = 0;
    while (n < max_iter) {
      ++n;
      
      stiff_tot = calc_stiffness(stiffnesses, strain, vals);
      szz = (stiff_tot * strain)(2);

      // Stop here if the stress is already close enough to 0
      if (abs(szz) < stop_crit) return strain(2);

      // Use a fixed-point method to compute the optimal strain value
      if (stiff_tot(2, 2) != 0.0) {
        strain(2) = -(stiff_tot(2, 0) * strain(0) + 
                      stiff_tot(2, 1) * strain(1) + 
                      stiff_tot(2, 3) * strain(3)) / stiff_tot(2, 2);
      }
      // In case the stiffness associated with zz is zero, nothing can be done
      else {
        return stiff_tot(2, 0) * strain(0) + stiff_tot(2, 1) * strain(1) 
            + stiff_tot(2, 3) * strain(3);
      }
      
    }
    
    return strain(2);
    
  }
  
}

#endif /* LIB_KELVINDECOMPOSITION_HXX */
