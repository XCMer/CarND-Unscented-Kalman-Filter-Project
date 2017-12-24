#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  cout << "GROUND: " << ground_truth[ground_truth.size() - 1] << endl;
  cout << "EST: " << estimations[estimations.size() - 1] << endl;

  // Accumulate squared residuals
  VectorXd residual = VectorXd::Zero(4);
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    VectorXd diff2 = (diff.array() * diff.array());
    residual += diff2;
  }

  // Calculate the mean
  VectorXd residual_mean = residual / estimations.size();

  // Calculate the squared root
  VectorXd rmse = residual_mean.array().sqrt();

  return rmse;
}