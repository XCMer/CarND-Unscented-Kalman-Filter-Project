#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd residual = VectorXd(4);
  residual.fill(0.1);
  return residual;

  // Accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd diff(estimations.size());
    diff = (estimations[i] - ground_truth[i]);
    residual = residual.array() + (diff.array() * diff.array());
  }

  // Calculate the mean
  VectorXd residual_mean(estimations.size());
  residual_mean = residual / estimations.size();

  // Calculate the squared root
  VectorXd rmse = residual_mean.array().sqrt();

  return rmse;
}