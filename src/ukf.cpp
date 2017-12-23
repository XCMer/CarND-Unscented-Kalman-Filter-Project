#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  n_x_ = 5;
  n_aug_ = n_x_ + 2;

  x_ = VectorXd(n_x_);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0.0);
  for (int i = 0; i < n_x_; i++) {
    P_(i, i) = 1;
  }

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  // Spreading parameter
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0);

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0);

  is_initialized_ = false;

  // Test functions
  TestGenerateSigmaPoints();
  TestPredictSigmaPoints();
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == meas_package.LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

      // Duplicated because individual sensors can be turned off
      is_initialized_ = true;
      last_timestamp_ = meas_package.timestamp_;
    } else {
      double r = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      x_ << (r * std::cos(theta)), (r * std::sin(theta)), 0, 0, 0;

      // Duplicated because individual sensors can be turned off
      is_initialized_ = true;
      last_timestamp_ = meas_package.timestamp_;
    }
  } else {
    if ((meas_package.sensor_type_ == meas_package.LASER) && use_laser_) {
      // Duplicated since a sensor can be turned off
      double delta_t = (meas_package.timestamp_ - last_timestamp_) / 10e6;
      last_timestamp_ = meas_package.timestamp_;

      Prediction(delta_t);
      UpdateLidar(meas_package);

    } else if ((meas_package.sensor_type_ == meas_package.RADAR) && use_radar_) {
      // Duplicated since a sensor can be turned off
      double delta_t = (meas_package.timestamp_ - last_timestamp_) / 10e6;
      last_timestamp_ = meas_package.timestamp_;

      Prediction(delta_t);
      UpdateRadar(meas_package);
    }
  }

//  cout << x_ << endl;
//  cout << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // First, we generate the sigma points from the current state
  MatrixXd Xsig_aug = GenerateSigmaPoints(x_, P_, std_a_, std_yawdd_);
  cout << "Sigma points" << endl;
  cout << Xsig_aug << endl;

  // Then, we get the predicted sigma points
  Xsig_pred_ = SigmaPointPrediction(Xsig_aug, delta_t);
  cout << "Sigma points prediction" << endl;
  cout << Xsig_pred_ << endl;

  // Finally, we update the state mean and covariance with the prediction
  // convert from sigma points back to state variable form
  cout << x_ << endl;
  cout << P_ << endl;
  PredictMeanAndCovariance(Xsig_pred_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Calculate cross correlation matrix (Tc)
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd Tc = MatrixXd(n_x_, n_radar_);

  // Get predicted value for Z
  MatrixXd Zsig = SigmaPointsToRadarMeasurement();

  VectorXd z_pred = VectorXd(n_radar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance (S)
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise
  MatrixXd R = MatrixXd(n_radar_, n_radar_);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  // For 2n+1 sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

/**
 * Generates sigma points for the state
 * @return Sigma point matrix
 */
MatrixXd UKF::GenerateSigmaPoints(const VectorXd &x, const MatrixXd &P, const double std_a, const double std_yawdd) {
  // Augmentation
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  P_aug(n_x_, n_x_) = std_a * std_a;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd * std_yawdd;

  // Sigma-point matrix
  MatrixXd Xsig = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Square root of P
  MatrixXd A = P_aug.llt().matrixL();
  cout << "A" << endl;
  cout << A << endl;

  // First column is the mean, so the same as x
  Xsig.col(0) = x_aug;

  // Calculate sigma points
  // Every calculation is a column in Xsig
  for (int i = 0; i < n_aug_; i++) {
    Xsig.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig.col(n_aug_ + i + 1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  return Xsig;
}


/**
 * Given sigma points and the time delta, gives us
 * the predicted sigma points for the current time.
 * @param Xsig_aug Sigma points from the last state
 * @param delta_t Time elapsed since the last measurement
 * @return Predicted sigma point matrix
 */
MatrixXd UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // Extract values for readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values
    double px_p, py_p;

    // Avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    // Remaining predicted state values
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  // Return the predicted sigma points
  return Xsig_pred;
}


/**
 * Updates the state and its covariance given the predicted sigma points
 *
 * @param Xsig_pred The predicted sigma points at the current time
 */
void UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred) {
  // Calculate weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  // Predicted state mean from weights and predicted sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  // Predicted state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    // Angle normalization
    cout << "ANGLE " << x_diff(3) << endl;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}


MatrixXd UKF::SigmaPointsToRadarMeasurement() {
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);

  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Extract values
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); // r
    Zsig(1, i) = atan2(p_y, p_x); // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  return Zsig;
}

void UKF::TestGenerateSigmaPoints() {
  // Example state
  VectorXd x = VectorXd(n_x_);
  x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  // Example covariance
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
          0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  MatrixXd Xsig = GenerateSigmaPoints(x, P, 0.2, 0.2);


  MatrixXd Xsig_expected = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_expected
          << 5.7441, 5.85768, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.63052, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441,
          1.38, 1.34566, 1.52806, 1.38, 1.38, 1.38, 1.38, 1.38, 1.41434, 1.23194, 1.38, 1.38, 1.38, 1.38, 1.38,
          2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.2049, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.2049, 2.2049,
          0.5015, 0.44339, 0.631886, 0.516923, 0.595227, 0.5015, 0.5015, 0.5015, 0.55961, 0.371114, 0.486077, 0.407773, 0.5015, 0.5015, 0.5015,
          0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.3528, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.3528,
          0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641, 0,
          0, 0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641;

  cout << "TestGenerateSigmaPoints" << endl;
  CompareMatrix(Xsig_expected, Xsig);
}

void UKF::TestPredictSigmaPoints() {
  // Sample augmented matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug <<
           5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
          1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
          2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
          0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
          0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
          0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
          0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  MatrixXd Xsig_pred = SigmaPointPrediction(Xsig_aug, 0.1);

  MatrixXd Xsig_pred_expected = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_expected
          << 5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
          1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
          2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
          0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
          0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;

  cout << "TestPredictSigmaPoints" << endl;
  CompareMatrix(Xsig_pred_expected, Xsig_pred);
}

void UKF::TestPredictMeanAndCovariance() {

}

void UKF::CompareMatrix(const MatrixXd &expected, const MatrixXd &predicted) {
  cout << "Expected" << endl;
  cout << expected << endl;

  cout << "Predicted" << endl;
  cout << predicted << endl;

  for (int i = 0; i < expected.rows(); i++) {
    for (int j = 0; j < expected.cols(); j++) {
      double diff = abs(expected(i, j) - predicted(i, j));
      if (diff >= 0.001) {
        cout << "Difference found A(" << i << "," << j << ")=" << expected(i, j)
             << " B(" << i << "," << j << ")=" << predicted(i, j)
             << " Diff=" << diff << endl;

        assert(diff < 0.001);
      }
    }
  }
}
