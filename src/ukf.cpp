#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <utility>

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

  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

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

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  is_initialized_ = false;

  // Test functions
  TestGenerateSigmaPoints();
  TestPredictSigmaPoints();
  TestPredictMeanAndCovariance();
  TestPredictRadar();
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
      cout << "LIDAR Predict x: " << endl;
      cout << x_ << endl;

      cout << "LIDAR Predict P: " << endl;
      cout << P_ << endl;

      State state = UpdateLidar(meas_package, x_, P_, Xsig_pred_, weights_, std_laspx_, std_laspy_);
      x_ = state.x;
      P_ = state.P;

      cout << "LIDAR Update x: " << endl;
      cout << x_ << endl;

      cout << "LIDAR Update P: " << endl;
      cout << P_ << endl;

    } else if ((meas_package.sensor_type_ == meas_package.RADAR) && use_radar_) {
      // Duplicated since a sensor can be turned off
      double delta_t = (meas_package.timestamp_ - last_timestamp_) / 10e6;
      last_timestamp_ = meas_package.timestamp_;

      Prediction(delta_t);
      cout << "RADAR Predict x: " << endl;
      cout << x_ << endl;

      cout << "RADAR Predict P: " << endl;
      cout << P_ << endl;

      State state = UpdateRadar(meas_package, x_, P_, Xsig_pred_, weights_, std_radr_, std_radphi_, std_radrd_);
      x_ = state.x;
      P_ = state.P;

      cout << "RADAR Update x: " << endl;
      cout << x_ << endl;

      cout << "RADAR Update P: " << endl;
      cout << P_ << endl;
    }
  }
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
//  cout << "Sigma points" << endl;
//  cout << Xsig_aug << endl;

  // Then, we get the predicted sigma points
  Xsig_pred_ = SigmaPointPrediction(Xsig_aug, delta_t);
//  cout << "Sigma points prediction" << endl;
//  cout << Xsig_pred_ << endl;

  // Finally, we update the state mean and covariance with the prediction
  // convert from sigma points back to state variable form
  State state = PredictMeanAndCovariance(Xsig_pred_);
  x_ = state.x;
  P_ = state.P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
State UKF::UpdateLidar(MeasurementPackage meas_package, VectorXd x, MatrixXd P, MatrixXd Xsig_pred,
                      VectorXd weights, double std_laspx, double std_laspy) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  VectorXd z = meas_package.raw_measurements_;

  // Predicted value in measurement space
  MatrixXd Zsig = SigmaPointsToLidarMeasurement(Xsig_pred);

  VectorXd z_pred = VectorXd::Zero(n_lidar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  // S
  MatrixXd S = MatrixXd::Zero(n_lidar_, n_lidar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise
  MatrixXd R = MatrixXd(n_lidar_, n_lidar_);
  R << std_laspx * std_laspx, 0,
       0, std_laspy * std_laspy;
  S = S + R;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_lidar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K * S * K.transpose();

  State state = State();
  state.x = x;
  state.P = P;

  return state;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
State UKF::UpdateRadar(MeasurementPackage meas_package, VectorXd x, MatrixXd P, MatrixXd Xsig_pred,
                       VectorXd weights, double std_radr, double std_radphi, double std_radrd) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  VectorXd z = meas_package.raw_measurements_;

  // Get predicted value for Z
  MatrixXd Zsig = SigmaPointsToRadarMeasurement(Xsig_pred);
//  cout << "ZIG" << endl;
//  cout << Zsig << endl;

  VectorXd z_pred = VectorXd::Zero(n_radar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  VectorXd x_pred = VectorXd::Zero(n_x_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_pred = x_pred + weights(i) * Xsig_pred.col(i);
  }
//  cout << "z_pred" << endl;
//  cout << z_pred << endl;

  // Measurement covariance (S)
  MatrixXd S = MatrixXd::Zero(n_radar_, n_radar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }
//  cout << "S" << endl;
//  cout << S << endl;

  // Add measurement noise
  MatrixXd R = MatrixXd(n_radar_, n_radar_);
  R << std_radr * std_radr, 0, 0,
       0, std_radphi * std_radphi, 0,
       0, 0, std_radrd * std_radrd;
  S = S + R;
//  cout << "S+R" << endl;
//  cout << S << endl;

  // Calculate cross correlation matrix (Tc)
  // For 2n+1 sigma points
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_radar_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x_pred;

    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
//  cout << "Tc" << endl;
//  cout << Tc << endl;

  // Residual
  VectorXd z_diff = z - z_pred;

  // Angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // Update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K * S * K.transpose();

  State state = State();
  state.x = x;
  state.P = P;

  cout << "NIS= " << (z_diff.transpose() * S.inverse() * z_diff) << endl;

  return state;
}

/**
 * Generates sigma points for the state
 * @return Sigma point matrix
 */
MatrixXd UKF::GenerateSigmaPoints(const VectorXd &x, const MatrixXd &P, const double std_a, const double std_yawdd) {
  // Augmentation
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  P_aug(n_x_, n_x_) = std_a * std_a;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd * std_yawdd;

  // Sigma-point matrix
  MatrixXd Xsig = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  // Square root of P
  MatrixXd A = P_aug.llt().matrixL();
//  cout << "A" << endl;
//  cout << A << endl;

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
  MatrixXd Xsig_pred = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

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
State UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred) {
  State state;

  // Calculate weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  // Predicted state mean from weights and predicted sigma points
  VectorXd x = VectorXd::Zero(n_x_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x + weights_(i) * Xsig_pred.col(i);
  }

  // Predicted state covariance matrix
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x;

    // Angle normalization
    cout << "ANGLE " << x_diff(3) << endl;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  state.x = x;
  state.P = P;

  return state;
}


MatrixXd UKF::SigmaPointsToRadarMeasurement(MatrixXd Xsig_pred) {
  MatrixXd Zsig = MatrixXd::Zero(n_radar_, 2 * n_aug_ + 1);

  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Extract values
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); // r
    Zsig(1, i) = atan2(p_y, p_x); // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  return Zsig;
}

MatrixXd UKF::SigmaPointsToLidarMeasurement(MatrixXd Xsig_pred) {
  MatrixXd Zsig = MatrixXd::Zero(n_lidar_, 2 * n_aug_ + 1);

  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Extract values
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);

    // Measurement model
    Zsig(0, i) = p_x; // x position
    Zsig(1, i) = p_y; // y position
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
  // Sample predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred <<
            5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
          1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
          0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  State state_pred = PredictMeanAndCovariance(Xsig_pred);

  VectorXd x_expected = VectorXd(n_x_);
  x_expected << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  MatrixXd P_expected = MatrixXd(n_x_, n_x_);
  P_expected << 0.00543425, -0.0024053, 0.00341576, -0.00348196, -0.00299378,
          -0.0024053, 0.010845, 0.0014923, 0.00980182, 0.00791091,
          0.00341576, 0.0014923, 0.00580129, 0.000778632, 0.000792973,
          -0.00348196, 0.00980182, 0.000778632, 0.0119238, 0.0112491,
          -0.00299378, 0.00791091, 0.000792973, 0.0112491, 0.0126972;

  cout << "TestPredictMeanAndCovariance x" << endl;
  CompareVector(x_expected, state_pred.x);

  cout << "TestPredictMeanAndCovariance P" << endl;
  CompareMatrix(P_expected, state_pred.P);
}

void UKF::TestPredictRadar() {
  // Define vector for weights
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  weights(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    weights(i) = 0.5 / (n_aug_ + lambda_);
  }

  // Example predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred
          << 5.9374, 6.0640, 5.925, 5.9436, 5.9266, 5.9374, 5.9389, 5.9374, 5.8106, 5.9457, 5.9310, 5.9465, 5.9374, 5.9359, 5.93744,
          1.48, 1.4436, 1.660, 1.4934, 1.5036, 1.48, 1.4868, 1.48, 1.5271, 1.3104, 1.4787, 1.4674, 1.48, 1.4851, 1.486,
          2.204, 2.2841, 2.2455, 2.2958, 2.204, 2.204, 2.2395, 2.204, 2.1256, 2.1642, 2.1139, 2.204, 2.204, 2.1702, 2.2049,
          0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337, 0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188, 0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633, 0.4841, 0.41872, 0.352, 0.38744, 0.40562, 0.24347, 0.32926, 0.2214, 0.28687, 0.352, 0.318159;

  // Example state vector
  VectorXd x = VectorXd(n_x_);
  x << 5.93637,
       1.49035,
       2.20528,
       0.536853,
       0.353577;

  // Exaple state covar
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P << 0.0054342, -0.002405, 0.0034157, -0.0034819, -0.00299378,
          -0.002405, 0.01084, 0.001492, 0.0098018, 0.00791091,
          0.0034157, 0.001492, 0.0058012, 0.00077863, 0.000792973,
          -0.0034819, 0.0098018, 0.00077863, 0.011923, 0.0112491,
          -0.0029937, 0.0079109, 0.00079297, 0.011249, 0.0126972;

  // Create sigma points for measurement space
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);
  Zsig << 6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
          0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
          2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

  // Example predicted measurement mean
  VectorXd z_pred = VectorXd(n_radar_);
  z_pred <<
         6.12155,
          0.245993,
          2.10313;

  // Example predicted measurement covar
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
  S << 0.0946171, -0.000139448, 0.00407016,
       -0.000139448, 0.000617548, -0.000770652,
       0.00407016, -0.000770652, 0.0180917;

  // Example incoming radar measurement
  VectorXd z = VectorXd(n_radar_);
  z << 5.9214,
       0.2187,
       2.0062;

  MeasurementPackage mp = MeasurementPackage();
  mp.raw_measurements_ = z;
  State state = UpdateRadar(mp, x, P, Xsig_pred, weights, 0.3, 0.0175, 0.1);

  // Expected values
  VectorXd x_expected = VectorXd(n_x_);
  x_expected << 5.92276,
          1.41823,
          2.15593,
          0.489274,
          0.321338;

  MatrixXd P_expected = MatrixXd(n_x_, n_x_);
  P_expected << 0.00361579, -0.000357881, 0.00208316, -0.000937196, -0.00071727,
          -0.000357881, 0.00539867, 0.00156846, 0.00455342, 0.00358885,
          0.00208316, 0.00156846, 0.00410651, 0.00160333, 0.00171811,
          -0.000937196, 0.00455342, 0.00160333, 0.00652634, 0.00669436,
          -0.00071719, 0.00358884, 0.00171811, 0.00669426, 0.00881797;

  /**
   * Expected Tc
   *  0.00468603 -0.000596985   0.00509758
   *  0.000295054   0.00181452  -0.00356734
   *  0.00367483  0.000100885   0.00512793
   * -0.000997283   0.00169207  -0.00571966
   * -0.000983935   0.00137287  -0.00547673
   */

  // Assertions
  cout << "TestPredictRadar x" << endl;
  CompareVector(x_expected, state.x);

  cout << "TestPredictRadar P" << endl;
  CompareMatrix(P_expected, state.P);
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

void UKF::CompareVector(const VectorXd &expected, const VectorXd &predicted) {
  cout << "Expected" << endl;
  cout << expected << endl;

  cout << "Predicted" << endl;
  cout << predicted << endl;

  for (int i = 0; i < expected.cols(); i++) {
    double diff = abs(expected(i) - predicted(i));
    if (diff >= 0.001) {
      cout << "Difference found A(" << i << ")=" << expected(i)
           << " B(" << i << ")=" << predicted(i)
           << " Diff=" << diff << endl;

      assert(diff < 0.001);
    }
  }
}
