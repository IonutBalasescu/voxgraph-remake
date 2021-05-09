#include "voxgraph/backend/constraint/cost_functions/registration_cost_function.h"

#include <utility>

#include <minkindr_conversions/kindr_tf.h>
#include <voxblox/interpolator/interpolator.h>

#include "voxgraph/frontend/submap_collection/voxgraph_submap.h"
#include "voxgraph/tools/tf_helper.h"

#include<iostream>

namespace voxgraph {
RegistrationCostFunction::RegistrationCostFunction(
    VoxgraphSubmap::ConstPtr reference_submap_ptr,
    VoxgraphSubmap::ConstPtr reading_submap_ptr,
    const RegistrationCostFunction::Config& config)
    : reading_submap_ptr_(reading_submap_ptr),
      reading_tsdf_layer_(reading_submap_ptr->getTsdfMap().getTsdfLayer()),
      reading_esdf_layer_(reading_submap_ptr->getEsdfMap().getEsdfLayer()),
      tsdf_interpolator_(&reading_submap_ptr->getTsdfMap().getTsdfLayer()),
      esdf_interpolator_(&reading_submap_ptr->getEsdfMap().getEsdfLayer()),
      registration_points_(reference_submap_ptr->getRegistrationPoints(
          config.registration_point_type)),
      config_(config) {
  //stat of original code
  //this part ensures the constraints
  // Ensure that the reference and reading submaps have gravity aligned Z-axes
  voxblox::Transformation::Vector6 T_i =
      reference_submap_ptr->getPose().log();
  voxblox::Transformation::Vector6 T_j =
      reading_submap_ptr->getPose().log();
  CHECK(T_i[3] < 1e-6 && T_i[4] < 1e-6)
      << "Submap Z axes should be gravity aligned, yet submap "
      << reference_submap_ptr->getID() << " had non-zero roll & pitch: ["
      << T_i[3] << ", " << T_i[4] << "]";
  CHECK(T_j[3] < 1e-6 && T_j[4] < 1e-6)
      << "Submap Z axes should be gravity aligned, yet submap "
      << reading_submap_ptr_->getID() << " had non-zero roll & pitch: ["
      << T_j[3] << ", " << T_j[4] << "]";

  // Set number of parameters: namely 2 poses, each having 4 params
  // (X,Y,Z,Yaw)
  mutable_parameter_block_sizes()->clear();
  mutable_parameter_block_sizes()->push_back(4);
  mutable_parameter_block_sizes()->push_back(4);

  // Set number of residuals
  int num_registration_residuals;
  if (config_.sampling_ratio != -1) {
    // Up/down sample the reference submap's registration points
    // according to the sampling ratio
    num_registration_residuals =
        static_cast<int>(config_.sampling_ratio * registration_points_.size());
  } else {
    // Deterministically use all registration points
    num_registration_residuals = registration_points_.size();
  }
  set_num_residuals(num_registration_residuals);
  //end of original code
}

// The matrix from the article (the aproximation with interpolation g B h)
const voxblox::InterpTable B =
    (voxblox::InterpTable() << 1,  0,  0,  0,  0,  0,  0,  0,
                              -1,  0,  0,  0,  1,  0,  0,  0,
                              -1,  0,  1,  0,  0,  0,  0,  0,
                              -1,  1,  0,  0,  0,  0,  0,  0,
                               1,  0, -1,  0, -1,  0,  1,  0,
                               1, -1, -1,  1,  0,  0,  0,  0,
                               1, -1,  0,  0, -1,  1,  0,  0,
                              -1,  1,  1, -1,  1, -1, -1,  1).finished();
// the appliction uses ceres; ceres is a library that offers the posibility
// to solve linear and nonlinear diferential equations
// parameters contains the values of each variable
// residual contains the residuals
// jacobian is computed if != null (this is specified in the dicumentation for ceres)
// http://ceres-solver.org/nnls_modeling.html
// the project is build in this manner so we are going to keep the structure
bool RegistrationCostFunction::Evaluate(double const* const* parameters,
                                        double* residuals,
                                        double** jacobians) const {
  //std::cout << "intra evaluate";
  
  unsigned int residual_idx = 0;
  double summed_weight = 0;

  // Get the number of parameters (used when addressing the jacobians array)
  CHECK_EQ(parameter_block_sizes()[0], parameter_block_sizes()[1]);
  int num_params = parameter_block_sizes()[0];

  // get the coordinate of the frames
  // and set the values to compute the transformation matrix
  voxblox::Transformation::Vector6 T_i;
  T_i[0] = parameters[0][0];  // x
  T_i[1] = parameters[0][1];  // y
  T_i[2] = parameters[0][2];  // z
  T_i[3] = 0;
  T_i[4] = 0;
  T_i[5] = parameters[0][3];  // yaw


  // the function exp receives an 6 element vector and return
  // the transformation matrix
  // gets(x y z 0 0 yaw)
  // return
  // (cos yaw) (-sin yaw) 0 x
  // (sin yaw) (cos yaw)  0 y
  //  0         0         1 z
  //  0         0         0 1
  // the form of the transformation matrix is also mentioned
  // in section 3.1 of this paper (quoted by the article)
  // https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
  const voxblox::Transformation S_i = voxblox::Transformation::exp(T_i);

  // The same thing is now done for the second frame too
  voxblox::Transformation::Vector6 T_j;
  T_j[0] = parameters[1][0];  // x 
  T_j[1] = parameters[1][1];  // y 
  T_j[2] = parameters[1][2];  // z 
  T_j[3] = 0;
  T_j[4] = 0;
  T_j[5] = parameters[1][3];  // yaw
  const voxblox::Transformation S_j = voxblox::Transformation::exp(T_j);

  // Compute the T_i_j matrix specified in the article
  const voxblox::Transformation T_j_i =
      S_j.inverse() * S_i;

  // Iterate the points from the pointcloud
  for (size_t sample_i = 0; sample_i < num_residuals(); sample_i++) {
    RegistrationPoint point;
    // This condition is from the original code
    // In various circumstances the function is either called
    // and evaluated on all the points and sampling_ratio
    // is set to -1 (by default, it cancels the sampling)
    // or it is set to a certain value and the point is chosen randomly
    if (config_.sampling_ratio == -1) {
      point = registration_points_[sample_i];
    } else {
      point = registration_points_.getRandomItem();
      point.weight = 1;
    }

    summed_weight += point.weight;
    voxblox::Point reference_coordinate = point.position;

    // Get distances and q_vector in reading submap
    const voxblox::Point reading_coordinate = T_j_i * reference_coordinate;
    bool interp_possible;
    voxblox::InterpVector distances;
    voxblox::InterpVector q_vector;

    
    // that is going to be used for further computations
    // again, from different parts of the code the functions is called with
    // different parameters
    // these 2 cases are from the original code
    // the purpose is to get the distance vector (mentioned in the article)
    // that is going to be used for further computations
    if (config_.use_esdf_distance) {
      const voxblox::EsdfVoxel* neighboring_voxels[8];
      interp_possible = esdf_interpolator_.getVoxelsAndQVector(
          reading_coordinate, neighboring_voxels, &q_vector);
      if (interp_possible) {
        for (int i = 0; i < distances.size(); ++i) {
          distances[i] = static_cast<voxblox::FloatingPoint>(
              neighboring_voxels[i]->distance);
        }
      }
    } else {
      const voxblox::TsdfVoxel* neighboring_voxels[8];
      interp_possible = tsdf_interpolator_.getVoxelsAndQVector(
          reading_coordinate, neighboring_voxels, &q_vector);
      if (interp_possible) {
        for (int i = 0; i < distances.size(); ++i) {
          distances[i] = static_cast<voxblox::FloatingPoint>(
              neighboring_voxels[i]->distance);
        }
      }
    }

    // Add residual
    if (interp_possible) {
      const double reading_distance = q_vector * (B * distances.transpose());
      residuals[residual_idx] = (point.distance - reading_distance) * point.weight;
    } else {
      residuals[residual_idx] = point.weight * config_.no_correspondence_cost;
    }


    // Save values usefull for jacobian
    float cos_e = std::cos(T_j[5]);
    float sin_e = std::sin(T_j[5]);
    // cos(yaw_reading - yaw_reference):
    float cos_emo = std::cos(T_j[5] - T_i[5]);
    // sin(yaw_reading - yaw_reference):
    float sin_emo = std::sin(T_j[5] - T_i[5]);
    float x_j = T_j[0];
    float y_j = T_j[1];
    float x_i = T_i[0];
    float y_i = T_i[1];


    // Jacobians if needed
    // if jacobians == null evaluate should not compute it
    if (jacobians != nullptr) {
      Eigen::Matrix<float, 1, 4> residual_i, residual_j;
      if (interp_possible) {
        // Calculate q_vector derivatives
        double inv = reading_tsdf_layer_.voxel_size_inv();
        double Dx = q_vector[1];
        double Dy = q_vector[2];
        double Dz = q_vector[3];
        
        // Build the Jacobian of the h vector
        // h = [1 dx dy dz dx*dy dy*dz dx*dz dx*dy*dz]
        //each column is the derivative over x y and z
        Eigen::Matrix<float, 8, 3> h_der;
        h_der << 0,             0,             0,
                 inv,           0,             0,
                 0,             inv,           0,
                 0,             0,             inv,
                 inv * Dy,      inv * Dx,      0,
                 0,             inv * Dz,      inv * Dy,
                 inv * Dz,      0,             inv * Dx,
                 inv * Dy * Dz, inv * Dx * Dz, inv * Dx * Dy;

        // Calculate the Jacobian of the interpolation function
        Eigen::Matrix<float, 1, 3> interpolation_derivative = distances * B.transpose() * h_der;

        float xi = reference_coordinate.x();
        float yi = reference_coordinate.y();

        // Jacobian of the transformation T_i_j * current_point
        Eigen::Matrix<float, 3, 4> T_der_1;
        T_der_1
            << cos_e, sin_e, 0, xi * sin_emo - yi * cos_emo,  // cos x       sin y 0  
              -sin_e, cos_e, 0, xi * cos_emo + yi * sin_emo,  // -sin x      cos y 0
               0,     0,     1, 0;                            // 0           0     1

        // Jacobian of the transformation T_i_j * current_point
        Eigen::Matrix<float, 3, 4> T_der_2;
        T_der_2
            << -cos_e, -sin_e, 0, -xi*sin_emo + yi*cos_emo + (x_j-x_i)*sin_e - (y_j-y_i)*cos_e,
                sin_e, -cos_e, 0, -xi*cos_emo - yi*sin_emo + (x_j-x_i)*cos_e + (y_j-y_i)*sin_e,
                0,      0,    -1,  0;

        // Jacobian of the residuals i pose params
        residual_i = -point.weight * interpolation_derivative * T_der_1;
        // Jacobian of the residuals j pose params
        residual_j = -point.weight * interpolation_derivative * T_der_2;

      } else {
        residual_i.setZero();
        residual_j.setZero();
      }

      // Store the Jacobians for Ceres
      if (jacobians[0] != nullptr) {
        for (int j = 0; j < 4; j++) {
          jacobians[0][residual_idx * 4 + j] = residual_i[j];
        }
      }
      if (jacobians[1] != nullptr) {
        for (int j = 0; j < 4; j++) {
          jacobians[1][residual_idx * 4 + j] = residual_j[j];
        }
      }
    }
    
    residual_idx++;
  }

  // Scale the residuals
  if (summed_weight == 0) return false;
  double factor = num_residuals() / summed_weight;
  for (int i = 0; i < num_residuals(); i++) {
    residuals[i] *= factor;
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        for (int j = 0; j < 4; j++) {
          jacobians[0][i * 4 + j] *= factor;
        }
      }
      if (jacobians[1] != nullptr) {
        for (int j = 0; j < 4; j++) {
          jacobians[1][i * 4 + j] *= factor;
        }
      }
    }
  }

  return true;
}
}  // namespace voxgraph
