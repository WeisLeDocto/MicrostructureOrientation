// g++ kelvin_lib.cpp -o kelvin_lib.so -shared -I /usr/include/eigen3 -fPIC -O3 -march=native -ffast-math

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <array>
#include <algorithm>

#include "kelvin_lib.hpp"

/**
* Performs the integration of the stiffness tensor multiplied by a Gaussian
* signal over the entire angle space.
*/
Eigen::Matrix<double, 6, 6> kelvin_integrated_tensor(double lambda_h, 
                                                     double lambda_1, 
                                                     double lambda_2, 
                                                     double lambda_3, 
                                                     double lambda_4, 
                                                     double lambda_5, 
                                                     double sigma,
                                                     double m,
                                                     bool skip_z = false) {
  
  Eigen::Matrix<double, 6, 6> integrated = Eigen::Matrix<double, 6, 6>::Zero();

  /// Pe-compute some terms to optimize speed and avoid redundancy
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

  /// Compute the terms on the upper half of the tensor
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

  /// Only compute the terms in the z direction if requested
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

  /// The tensor is symmetrical, so the lower half terms can just be copied
  integrated.triangularView<Eigen::Lower>() = integrated.transpose();
  
  return integrated;
  
}


/**
* Performs the integration of the stiffness tensor multiplied by a Gaussian
* signal over the entire angle space, in the case when the m factor is very
* close to 0.
*/
Eigen::Matrix<double, 6, 6> zero_m_approximation(double lambda_h, 
                                                 double lambda_1, 
                                                 double lambda_2, 
                                                 double lambda_3, 
                                                 double lambda_4, 
                                                 double lambda_5, 
                                                 double sigma,
                                                 double m,
                                                 bool skip_z = false) {
  
  Eigen::Matrix<double, 6, 6> integrated = Eigen::Matrix<double, 6, 6>::Zero();

  /// Pe-compute some terms to optimize speed and avoid redundancy
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
  const double quad_term1 = 19.0 * lam1_sq - 14.0 * lam1_lam2 + 11.0 * lam2_sq 
      - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;
  const double quad_term2 = 3.0 * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 
      * lam1_lam5 + 4.0 * lam2_lam5;
  const double quad_term3 = 9.0 * lam1_sq + 6.0 * lam1_lam2 + lam2_sq - 24.0 
      * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;
  const double quad_term4 = 15.0 * lam1_sq - 6.0 * lam1_lam2 + 7.0 * lam2_sq 
      - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq;

  /// Compute the terms on the upper half of the tensor
  integrated(0, 0) = 1.0 / 48.0 * (term_A * exp_8s + term_B * exp_6s + term_C) 
      * exp_neg8s;
  integrated(0, 1) = -1.0 / 48.0 * (term_D * exp_8s + term_C) * exp_neg8s;
  integrated(0, 2) = -1.0 / 12.0 * (term_E * exp_2s + term_G) * exp_neg2s;
  integrated(1, 1) = 1.0 / 48.0 * (term_A * exp_8s - term_B * exp_6s + term_C) 
      * exp_neg8s;
  integrated(1, 2) = -1.0 / 12.0 * (term_E * exp_2s - term_G) * exp_neg2s;
  integrated(2, 2) = 1.0 / 6.0 * lam1_l + 1.0 / 2.0 * lam2_l + 1.0 / 3.0 
      * lamh_l;
  integrated(3, 3) = 1.0 / 8.0 * (term_F * exp_8s - 3.0 * lam1_l - lam2_l 
      + 4.0 * lam5_l) * exp_neg8s;

  /// Only compute the terms in the z direction if requested
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

  /// The result is a sum of two terms, computing the second one here
  integrated(0, 0) += 1.0 / 256.0 * m * (quad_term1 * exp_16s +  4.0 
      * quad_term2 * exp_14s - 16.0 * cross_term * exp_12s + 6.0 * cross_term 
      * exp_8s - 4.0 * quad_term2 * exp_6s - quad_term3) * exp_neg16s;
  integrated(0, 1) += -1.0 / 256.0 * m * ((11.0 * lam1_sq + 2.0 * lam1_lam2 
      + 3.0 * lam2_sq - 24.0 * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq) 
      * exp_16s - 8.0 * cross_term * exp_12s + 6.0 * cross_term * exp_8s 
      - quad_term3) * exp_neg16s;
  integrated(0, 2) += -1.0 / 64.0 * m * (2.0 * cross_term * exp_10s + (3.0 
      * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 
      * lam2_lam5) * exp_8s - 2.0 * cross_term * exp_6s - (3.0 * lam1_sq - 2.0 
      * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 * lam2_lam5)) * exp_neg10s;
  integrated(1, 1) += 1.0 / 256.0 * m * (quad_term1 * exp_16s - 4.0 
      * quad_term2 * exp_14s - 16.0 * cross_term * exp_12s + 6.0 * cross_term 
      * exp_8s + 4.0 * quad_term2 * exp_6s - quad_term3) * exp_neg16s;
  integrated(1, 2) += -1.0 / 64.0 * m * (2.0 * cross_term * exp_10s - (3.0 
      * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 
      * lam2_lam5) * exp_8s - 2.0 * cross_term * exp_6s + (3.0 * lam1_sq - 2.0 
      * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5 + 4.0 * lam2_lam5)) * exp_neg10s;
  integrated(2, 2) += 1.0 / 16.0 * cross_term_m * (exp_4s - 1.0) * exp_neg4s;
  integrated(3, 3) += 1.0 / 128.0 * m * (quad_term4 * exp_16s - 6.0 
      * cross_term * exp_8s - quad_term3) * exp_neg16s;

  /// Only compute the terms in the z direction if requested
  if (!skip_z) {
    const double z_cross = lam3_l * lam3_l - 2.0 * lam3_l * lam4_l + lam4_l 
        * lam4_l;
    const double z_term = z_cross * m * (exp_4s - 1.0) * exp_neg4s;
    integrated(4, 4) += 0.125 * z_term;
    integrated(5, 5) += 0.125 * z_term;
  }

  /// The tensor is symmetrical, so the lower half terms can just be copied
  integrated.triangularView<Eigen::Lower>() = integrated.transpose();
  
  return integrated;
  
}


/**
* Applies a rotation of R3 around the z axis at the given angle to a tensor.
*/
Eigen::Matrix<double, 6, 6> rotate(const Eigen::Matrix<double, 6, 6> tensor, 
                                   double angle) {

  /// Build the rotation matrix in R3
  Eigen::Matrix<double, 3, 3> rot_mat =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  /// Build the rotation tensor from the rotation matrix
  Eigen::Matrix<double, 6, 6> rot_ten;
  
  rot_ten(0, 0) = rot_mat(0, 0) * rot_mat(0, 0);
  rot_ten(0, 1)= rot_mat(1, 0) * rot_mat(1, 0);
  rot_ten(0, 2) = rot_mat(2, 0) * rot_mat(2, 0);
  rot_ten(0, 3) = sqrt(2) * rot_mat(0, 0) * rot_mat(1, 0);
  rot_ten(0, 4) = sqrt(2) * rot_mat(0, 0) * rot_mat(2, 0);
  rot_ten(0, 5) = sqrt(2) * rot_mat(1, 0) * rot_mat(2, 0);

  rot_ten(1, 0) = rot_mat(0, 1) * rot_mat(0, 1);
  rot_ten(1, 1) = rot_mat(1, 1) * rot_mat(1, 1);
  rot_ten(1, 2) = rot_mat(2, 1) * rot_mat(2, 1);
  rot_ten(1, 3) = sqrt(2) * rot_mat(0, 1) * rot_mat(1, 1);
  rot_ten(1, 4) = sqrt(2) * rot_mat(0, 1) * rot_mat(2, 1);
  rot_ten(1, 5) = sqrt(2) * rot_mat(1, 1) * rot_mat(2, 1);

  rot_ten(2, 0) = rot_mat(0, 2) * rot_mat(0, 2);
  rot_ten(2, 1) = rot_mat(1, 2) * rot_mat(1, 2);
  rot_ten(2, 2) = rot_mat(2, 2) * rot_mat(2, 2);
  rot_ten(2, 3) = sqrt(2) * rot_mat(0, 2) * rot_mat(1, 2);
  rot_ten(2, 4) = sqrt(2) * rot_mat(0, 2) * rot_mat(2, 2);
  rot_ten(2, 5) = sqrt(2) * rot_mat(1, 2) * rot_mat(2, 2);

  rot_ten(3, 0) = sqrt(2) * rot_mat(0, 0) * rot_mat(0, 1);
  rot_ten(3, 1) = sqrt(2) * rot_mat(1, 0) * rot_mat(1, 1);
  rot_ten(3, 2) = sqrt(2) * rot_mat(2, 0) * rot_mat(2, 1);
  rot_ten(3, 3) = rot_mat(0, 0) * rot_mat(1, 1) + rot_mat(0, 1) 
      * rot_mat(1, 0);
  rot_ten(3, 4) = rot_mat(0, 0) * rot_mat(2, 1) + rot_mat(0, 1) 
      * rot_mat(2, 0);
  rot_ten(3, 5) = rot_mat(1, 0) * rot_mat(2, 1) + rot_mat(1, 1) 
      * rot_mat(2, 0);

  rot_ten(4, 0) = sqrt(2) * rot_mat(0, 0) * rot_mat(0, 2);
  rot_ten(4, 1) = sqrt(2) * rot_mat(1, 0) * rot_mat(1, 2);
  rot_ten(4, 2) = sqrt(2) * rot_mat(2, 0) * rot_mat(2, 2);
  rot_ten(4, 3) = rot_mat(0, 0) * rot_mat(1, 2) + rot_mat(0, 2) 
      * rot_mat(1, 0);
  rot_ten(4, 4) = rot_mat(0, 0) * rot_mat(2, 2) + rot_mat(0, 2) 
      * rot_mat(2, 0);
  rot_ten(4, 5) = rot_mat(1, 0) * rot_mat(2, 2) + rot_mat(1, 2) 
      * rot_mat(2, 0);

  rot_ten(5, 0) = sqrt(2) * rot_mat(0, 1) * rot_mat(0, 2);
  rot_ten(5, 1) = sqrt(2) * rot_mat(1, 1) * rot_mat(1, 2);
  rot_ten(5, 2) = sqrt(2) * rot_mat(2, 1) * rot_mat(2, 2);
  rot_ten(5, 3) = rot_mat(0, 1) * rot_mat(1, 2) + rot_mat(0, 2) 
      * rot_mat(1, 1);
  rot_ten(5, 4) = rot_mat(0, 1) * rot_mat(2, 2) + rot_mat(0, 2) 
      * rot_mat(2, 1);
  rot_ten(5, 5) = rot_mat(1, 1) * rot_mat(2, 2) + rot_mat(1, 2) 
      * rot_mat(2, 1);

  /// Finally, apply the rotation tensor
  return rot_ten.transpose() * tensor * rot_ten;

}


/**
* Computes the final stress value from the stiffness tensor for each order, the
* multiplicative factor for each order, and the strain tensor.
*/
Eigen::Matrix<double, 6, 1> final_stress(
    const std::array<Eigen::Matrix<double, 6, 6>, 5> &stiffness,
    const Eigen::Matrix<double, 6, 1> strain,
    const std::array<double, 5> &vals) {
    
  std::array<double, 5> factor = {1.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Matrix<double, 6, 6> stiff_tot = Eigen::Matrix<double, 6, 6>::Zero();

  /// Iterate over all the orders
  for (int j = 0; j < 5; j++) {
    /// Skip inactive orders
    if (vals[j] == 0.0) continue;

    /// For each order greater than 1, there is a multiplicative factor
    /// dependent on the strain and stiffness
    if (j > 0) {
      factor[j] = strain.dot(stiffness[j] * strain);
    }

    // The total equivalent stiffness is the sum of the ones for each order
    stiff_tot += vals[j] * (j + 1) * pow(factor[j], j) * stiffness[j];
  }
  
  return stiff_tot * strain;
  
}


/**
* Determines the value of the zz strain so that the zz stress is close enough
* to 0, to be in a plane stress configuration.
*/
float calc_ezz_plane_stress(
    const std::array<Eigen::Matrix<double, 6, 6>, 5> &stiffness,
    Eigen::Matrix<double, 6, 1> strain,
    const std::array<double, 5> &vals,
    const double stop_crit,
    const int max_iter) {
  
  std::array<double, 5> factor = {1.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Matrix<double, 1, 6> stiff_tot = Eigen::Matrix<double, 1, 6>::Zero();

  /// Iterate over all the orders
  for (int j = 0; j < 5; j++) {
    /// Skip inactive orders
    if (vals[j] == 0.0) continue;
    /// For each order greater than 1, there is a multiplicative factor
    /// dependent on the strain and stiffness
    if (j > 0) {
      factor[j] = strain.dot(stiffness[j] * strain);
    }
    /// The total equivalent stiffness is the sum of the ones for each order
    stiff_tot += vals[j] * (j + 1) * pow(factor[j], j) * stiffness[j].row(2);
  }

  /// Compute the zz strain component
  double szz = stiff_tot.dot(strain);

  /// Stop here if the stress is already close enough to 0
  if (abs(szz) < stop_crit) return strain(2, 0);

  /// Use a fixed-point method to compute the optimal strain value
  if (stiff_tot(2) != 0.0) {
    strain(2, 0) = -(stiff_tot(0) * strain(0, 0) + 
                     stiff_tot(1) * strain(1, 0) + 
                     stiff_tot(3) * strain(3, 0)) / stiff_tot(2);
  }
  /// In case the stiffness associated with zz is zero, nothing can be done
  else {
    return stiff_tot(0) * strain(0, 0) + stiff_tot(1) * strain(1, 0) 
        + stiff_tot(3) * strain(3, 0);
  }

  /// Iterate until reaching a solution or until reaching max iterations
  int n = 0;
  while (n < max_iter) {
    ++n;
    
    factor = {1.0, 0.0, 0.0, 0.0, 0.0};
    stiff_tot = Eigen::Matrix<double, 1, 6>::Zero();

    /// Iterate over all the orders
    for (int j = 0; j < 5; j++) {
      /// Skip inactive orders
      if (vals[j] == 0.0) continue;
      /// For each order greater than 1, there is a multiplicative factor
      /// dependent on the strain and stiffness
      if (j > 0) {
        factor[j] = strain.dot(stiffness[j] * strain);
      }
      /// The total equivalent stiffness is the sum of the ones for each order
      stiff_tot += vals[j] * (j + 1) * pow(factor[j], j) * stiffness[j].row(2);
    }

    /// Compute the zz strain component
    szz = stiff_tot.dot(strain);

    /// Stop here if the stress is already close enough to 0
    if (abs(szz) < stop_crit) return strain(2, 0);

    /// Use a fixed-point method to compute the optimal strain value
    if (stiff_tot(2) != 0.0) {
      strain(2, 0) = -(stiff_tot(0) * strain(0, 0) + 
                       stiff_tot(1) * strain(1, 0) + 
                       stiff_tot(3) * strain(3, 0)) / stiff_tot(2);
    }
    /// In case the stiffness associated with zz is zero, nothing can be done
    else {
      return stiff_tot(0) * strain(0, 0) + stiff_tot(1) * strain(1, 0) 
          + stiff_tot(3) * strain(3, 0);
    }
    
  }
  
  return strain(2, 0);
  
}


/**
* Given material parameters, microstructure parameters, order multiplicators,
* and a strain tensor, computes the stress tensor under the plane stress
* hypothesis.
*/
void calc_stress(double exx,
                 double eyy,
                 double exy,
                 double lamh,
                 double lam11,
                 double lam21,
                 double lam31,
                 double lam41,
                 double lam51,
                 double lam12,
                 double lam22,
                 double lam32,
                 double lam42,
                 double lam52,
                 double lam13,
                 double lam23,
                 double lam33,
                 double lam43,
                 double lam53,
                 double lam14,
                 double lam24,
                 double lam34,
                 double lam44,
                 double lam54,
                 double lam15,
                 double lam25,
                 double lam35,
                 double lam45,
                 double lam55,
                 double val1,
                 double val2,
                 double val3,
                 double val4,
                 double val5,
                 double theta_1,
                 double theta_2,
                 double theta_3,
                 double sigma_1,
                 double sigma_2,
                 double sigma_3,
                 double density,
                 double* sxx,
                 double* syy,
                 double* sxy) {

  /// Compute eigenvalues of the strain tensor
  const double trace = exx + eyy;
  const double sqrt_term = sqrt((exx - eyy) * (exx - eyy) + 4.0 * exy * exy);
  const double lambda1 = (trace + sqrt_term) * 0.5;
  const double lambda2 = (trace - sqrt_term) * 0.5;

  /// Compute the angle of the first eigenvector
  double theta_load;
  if (abs(exy) < 1e-12 && abs(exx - eyy) < 1e-12) {
      theta_load = 0.0;
  } else {
      theta_load = 0.5 * atan2(2.0 * exy, exx - eyy);
  }

  /// Organize the values in arrays for convenience
  const double sigstd[3] = {sigma_1, sigma_2, sigma_3};
  const double theta[3] = {theta_1, theta_2, theta_3};
  const std::array<double, 5> vals = {val1, val2, val3, val4, val5};
  const std::array<double, 5> lam1 = {lam11, lam12, lam13, lam14, lam15};
  const std::array<double, 5> lam2 = {lam21, lam22, lam23, lam24, lam25};
  const std::array<double, 5> lam3 = {lam31, lam32, lam33, lam34, lam35};
  const std::array<double, 5> lam4 = {lam41, lam42, lam43, lam44, lam45};
  const std::array<double, 5> lam5 = {lam51, lam52, lam53, lam54, lam55};

  /// Initialize the stiffness values with zeros
  std::array<Eigen::Matrix<double, 6, 6>, 5> stiffness;
  std::fill(stiffness.begin(), 
            stiffness.end(), 
            Eigen::Matrix<double, 6, 6>::Zero());
  Eigen::Matrix<double, 6, 6> homogenized;

  /// Iterate over up to three tissue layers
  int layer_count = 0;
  for (int i = 0; i < 3; i++) {

    /// A standard deviation too low indicates that a layer was not detected
    if (sigstd[i] < 0.01 && i > 0) break;

    /// Compute the integration factor as a function of the angle between the
    /// load and the fibers
    double m = cos(2.0 * (theta_load - theta[i]));
    /// Correct by the factor to account for anisotropy in the strain
    if (abs(lambda1) + abs(lambda2) > 1.0e-12) {
      m *= (abs(lambda1 - lambda2)) / (abs(lambda1) + abs(lambda2));
    }

    /// Iterate over the 5 orders of the model
    for (int j = 0; j < 5; j++) {

      /// Skip inactive orders
      if (vals[j] == 0.0) continue;

      /// Use the relevant function for computing the homogenized tensor,
      /// depending on the value of m
      if (abs(m) < 0.01) {
        homogenized = zero_m_approximation(
          lamh, lam1[j], lam2[j], lam3[j], lam4[j], lam5[j], sigstd[i],
          m, true).exp();
      }
      else {
        homogenized = kelvin_integrated_tensor(
          lamh, lam1[j], lam2[j], lam3[j], lam4[j], lam5[j], sigstd[i],
          m, true).pow(1.0 / m);
      }

      /// Rotate the homogenized tensor to align it with the fibers
      if (i == 0) {
        stiffness[j] = rotate(homogenized, theta[i]);
      }
      else {
        stiffness[j] += rotate(homogenized, theta[i]);
      }
      
    }
    
    ++layer_count;
  }

  /// The final stiffness tensor is averaged over the detected layers
  for (int j = 0; j < 5; j++) {
    stiffness[j] = stiffness[j] / layer_count;
  }

  /// Need to consider 3D to compute the stress in plane stress hypothesis
  Eigen::Matrix<double, 6, 1> strain_3d;
  strain_3d(0, 0) = exx;
  strain_3d(1, 0) = eyy;
  strain_3d(3, 0) = exy;
  strain_3d(4, 0) = 0.0;
  strain_3d(5, 0) = 0.0;

  /// Calculate the zz strain value that satisfies the plane stress hypothesis
  strain_3d(2, 0) = 0.0;
  strain_3d(2, 0) = calc_ezz_plane_stress(
      stiffness, 
      strain_3d, 
      vals, 
      std::max(exx, std::max(eyy, exy)) / 1000.0,
      100);

  /// Finally, compute the final 3D stress tensor
  const Eigen::Matrix<double, 6, 1> sig_3d = density * final_stress(stiffness,
                                                                    strain_3d,
                                                                    vals);
  
  *sxx = sig_3d(0, 0);
  *syy = sig_3d(1, 0);
  *sxy = sig_3d(3, 0);

}


void calc_stresses(double* exx,
                   double* eyy,
                   double* exy,
                   double lamh,
                   double lam11,
                   double lam21,
                   double lam31,
                   double lam41,
                   double lam51,
                   double lam12,
                   double lam22,
                   double lam32,
                   double lam42,
                   double lam52,
                   double lam13,
                   double lam23,
                   double lam33,
                   double lam43,
                   double lam53,
                   double lam14,
                   double lam24,
                   double lam34,
                   double lam44,
                   double lam54,
                   double lam15,
                   double lam25,
                   double lam35,
                   double lam45,
                   double lam55,
                   double val1,
                   double val2,
                   double val3,
                   double val4,
                   double val5,
                   double* theta_1,
                   double* theta_2,
                   double* theta_3,
                   double* sigma_1,
                   double* sigma_2,
                   double* sigma_3,
                   double* density,
                   int rows,
                   int cols,
                   double* sxx,
                   double* syy,
                   double* sxy) {

  int idx;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {

      idx = i * cols + j;
      calc_stress(exx[idx],
                  eyy[idx],
                  exy[idx],
                  lamh,
                  lam11,
                  lam21,
                  lam31,
                  lam41,
                  lam51,
                  lam12,
                  lam22,
                  lam32,
                  lam42,
                  lam52,
                  lam13,
                  lam23,
                  lam33,
                  lam43,
                  lam53,
                  lam14,
                  lam24,
                  lam34,
                  lam44,
                  lam54,
                  lam15,
                  lam25,
                  lam35,
                  lam45,
                  lam55,
                  val1,
                  val2,
                  val3,
                  val4,
                  val5,
                  theta_1[idx],
                  theta_2[idx],
                  theta_3[idx],
                  sigma_1[idx],
                  sigma_2[idx],
                  sigma_3[idx],
                  density[idx],
                  &sxx[idx],
                  &syy[idx],
                  &sxy[idx]);
    }
  }
}
