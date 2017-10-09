#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.05;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/7;

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

  // initial Normalized Innovation Squared (NIS)
  NIS_radar_ = 0;
  NIS_laser_ = 0;

  // log file names
  file_laser_ = "NIS_laser.txt";
  file_radar_ = "NIS_radar.txt";


  // empty log files
  ofstream fstream_laser_(file_laser_, ios::out|ios::trunc);
  if (!fstream_laser_.good()) {
    cerr << "file output error!" << endl;
  }
  else {
    fstream_laser_.close();
  }

  ofstream file2(file_radar_, ios::out|ios::trunc);
  if (!file2.good()) {
    cerr << "file output error!" << endl;
  }
  else {
    file2.close();
  }

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // Measurement noise covariance matrix R
  R_radar_ = MatrixXd(3, 3);
  R_laser_ = MatrixXd(2, 2);

  // Radar
  R_radar_.setZero();
  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;

  // Laser
  R_laser_.setZero();
  R_laser_(0, 0) = std_laspx_ * std_laspx_;
  R_laser_(1, 1) = std_laspy_ * std_laspy_;
  

  // Measurement Matrix H for Laser 
  // only px and py is projected to the measurement space
  H_laser_ = MatrixXd(2, 5);
  H_laser_.setZero();
  H_laser_(0, 0) = 1;
  H_laser_(1, 1) = 1;

  // Process noise covariance matrix Q_
  Q_ = MatrixXd(2, 2);
  Q_.setZero();
  Q_(0, 0) = std_a_ * std_a_;
  Q_(1, 1) = std_yawdd_ * std_yawdd_;

  // set state dimension
  n_x_ = 5;

  // set augmented dimension
  n_aug_ = 7;

  // define spreading parameter
  lambda_ = 3 - n_aug_;

  // define number of sigma points
  n_sig_ = 2 * n_aug_ +1;

  // set weights
  weights_ = VectorXd(n_sig_);
  weights_.fill(1 / (2* (lambda_ + n_aug_)));
  weights_(0) = lambda_/(lambda_ + n_aug_);

  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
  Xsig_pred_ = MatrixXd(n_aug_, n_sig_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.

  */
    /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "UKF: " << endl;
    // first measurement
    // set the 5D state vector to initial zeros
    x_.setZero();
    // set Identitiy for initial State covariance Matrix P_
    P_.setIdentity(n_x_, n_x_);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double p_x = meas_package.raw_measurements_(0)*cos(meas_package.raw_measurements_(1));
      double p_y = meas_package.raw_measurements_(0)*sin(meas_package.raw_measurements_(1)); 
      double v_x = meas_package.raw_measurements_(2)*cos(meas_package.raw_measurements_(1));
      double v_y = meas_package.raw_measurements_(2)*sin(meas_package.raw_measurements_(1)); 
      x_[0] = p_x; // px
      x_[1] = p_y;  // py
      x_[2] = sqrt(v_x*v_x + v_y*v_y); // nu
      x_[3] = 0.0; // psi
      x_[4] = 0.007; // psi_dot

      // initialize State Covariance Matrix P_ for Radar  --> Overfitting for test data
      //P_(0, 0) = 0.02;
      //P_(1, 1) = 0.02;
      //P_(2, 2) = 0.08;
      //P_(3, 3) = 1.55;
      //P_(4, 4) = 0.04;
    }
    
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state for Laser (cartesian)
      */
      x_[0] = meas_package.raw_measurements_(0);  // px
      x_[1] = meas_package.raw_measurements_(1);  // py
      x_[2] = 5.20; // nu
      x_[3] = 0.0; // psi
      x_[4] = 0.007; // psi_dot

      // initialize State Covariance Matrix P_ for Laser
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
      //P_(2, 2) = 0.08;
      //P_(3, 3) = 1.55;  //2.3945;
      //P_(4, 4) = 0.04; //0.1513;
    }  
      
    // set the previous timestamp
    time_us_ = meas_package.timestamp_;
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "Initialization successful" << endl;
    return;
  }

  //compute the time elapsed between the current and previous measurements

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  if (( dt > 0.001 ) && (use_laser_ || use_radar_)) {     //normal filter operation
    
    /*****************************************************************************
     *  Prediction & Update
     ****************************************************************************/

    if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_) {
      // Predict
      Prediction(dt);
      // Radar updates
      UpdateRadar(meas_package);
      cout << "Radar Update / NIS = " << NIS_radar_ << endl;
      cout << "x_ = " << x_ << endl;
      cout << "P_ = " << P_ << endl; 
    } 
    else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_) {
      // Predict
      Prediction(dt);
      // Laser updates
      UpdateLidar(meas_package);
      cout << "Laser Update / NIS = " << NIS_laser_ << endl;
      cout << "x_ = " << x_ << endl;
      cout << "P_ = " << P_ << endl; 
    }
  }

  else {
    // do nothing
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
  
  // Create Augmented Sigma Points
  AugmentedSigmaPoints(& Xsig_aug_);

  // Predict Sigma Points
  SigmaPointPrediction(delta_t, & Xsig_pred_);

  // Predict Mean and Covariance
  PredictMeanAndCovariance(& x_, & P_);
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

  //set measurement dimension, laser can measure p_x and p_y
  int n_z = 2;

  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  VectorXd y = z - H_laser_ * x_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;

  // update NIS radar
  NIS_laser_ = y.transpose() * Si * y;
  
  fstream_laser_.open(file_laser_, ios::out|ios::app);
  if (!fstream_laser_.good()) {
    cerr << "file output error!" << endl;
  }
  else {
    fstream_laser_ << meas_package.timestamp_ << ";" << NIS_laser_ << endl;
    fstream_laser_.close();
  }
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
  
  /*****************************************************************************
   *  Prediction Radar Measurement
   ****************************************************************************/

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    double c1 = p_x*p_x + p_y*p_y;
    double c2 = sqrt(c1);
    Zsig(0,i) = c2;                                             //r
    if (fabs(c1) < 0.0001) {
      cout << "Error - Division by zero!" << endl;
    }
    else {
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));


    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_radar_;

  /*****************************************************************************
  *  Update Radar
  ****************************************************************************/

  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // update NIS radar
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  fstream_radar_.open(file_radar_, ios::out|ios::app);
  if (!fstream_radar_.good()) {
    cerr << "file output error!" << endl;
  }
  else {
    fstream_radar_ << meas_package.timestamp_ << ";" << NIS_radar_ << endl;
    fstream_radar_.close();
  }
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
 
  //create augmented mean state
  x_aug.setZero(); 
  x_aug.head(5) = x_;
  
  //create augmented covariance matrix
  P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q_;
    
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  //create augmented sigma points
  
  //set first column of sigma points matrix
  Xsig_aug.col(0) = x_aug;
  
  //set remaining sigma points
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i);
      Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_ + n_aug_)*A.col(i);
  }

  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(double delta_t, MatrixXd* Xsig_out) {

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_);

  //predict sigma points
  for (int i = 0; i < n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }
 
  //write result
  *Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  VectorXd a = VectorXd(n_x_);

  
  //predict state mean
  x = Xsig_pred_ * weights_;
  
  //predict state covariance matrix
  P.setZero(5, 5);
  a.setZero();
  for (int i = 0; i < (n_sig_); i++) {
      a = (Xsig_pred_.col(i) - x);
      // angle normalization
      a(3) = NormalizeAngle(a(3));
      P = P + weights_(i) * a * a.transpose();
  }
 
  //write result
  *x_out = x;
  *P_out = P;
}

double UKF::NormalizeAngle(double angle)  {
  angle = atan2(sin(angle), cos(angle));
  return angle;
}