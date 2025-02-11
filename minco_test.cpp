// This source file shows an example of how to use the minco trajectory class
#include <iostream>

#include "minco.hpp"
#include "trajectory.hpp"

using namespace std;

int main()
{
    // Uniform MINCO with s = 3, m = 3, N = 4 and dT = 2
    minco::MINCO<3, 3, minco::Uniform> minco_uniform(4);
    // the start state. head.row(0) is the position, head.row(1) is the velocity, head.row(2) is the acceleration, etc.
    Eigen::Matrix<double, 3, 3> head;
    head.row(0) << 0, 0, 0;
    head.row(1) << 0, 0, 0;
    head.row(2) << 0, 0, 0;
    // the intermediate waypoints, each row is a waypoint.
    // because N = 4, there should be 3 waypoints.
    Eigen::Matrix<double, 3, 3> waypoints;
    waypoints.row(0) << 1, 0, 0;
    waypoints.row(1) << 1, 1, 0;
    waypoints.row(2) << 0, 1, 0;
    // the end state. tail.row(0) is the position, tail.row(1) is the velocity, tail.row(2) is the acceleration, etc.
    Eigen::Matrix<double, 3, 3> tail;
    tail.row(0) << 0, 1, 1;
    tail.row(1) << 0, 0, 0;
    tail.row(2) << 0, 0, 0;
    // generate the trajectory
    minco_uniform.setConditions(head);
    minco_uniform.setParameters(waypoints, tail, 2);
    // get the coefficients matrix of the trajectory
    cout << "Uniform MINCO coefficients: " << endl << minco_uniform.getCoeffs() << endl;
    // get the energy of the trajectory
    cout << "Uniform MINCO energy: " << minco_uniform.getEnergy() << endl;
    // get the energy partial derivatives by the coefficients
    cout << "Uniform MINCO energy partial derivatives: " << endl << minco_uniform.getEnergyPartialGradByCoeffs() << endl;
    // get the energy partial derivatives by dT
    cout << "Uniform MINCO energy partial derivatives by dT: " << minco_uniform.getEnergyPartialGradBydT() << endl;
    // construct the trajectory
    PolynominalTrajectory<3> uniform_trajectory(minco_uniform.getCoeffs(), Eigen::Vector3d(2, 2, 2));
    // get the position and velocity at time 2.5
    cout << "Uniform MINCO position at time 2.5: " << uniform_trajectory.at(2.5, 0).transpose() << endl;
    cout << "Uniform MINCO velocity at time 2.5: " << uniform_trajectory.at(2.5, 1).transpose() << endl;

    // NonUniform MINCO with s = 3, m = 2, N = 4 and T = [1, 2, 2, 1]
    minco::MINCO<3, 2, minco::NonUniform> minco_nonuniform(4);
    // the start state. head.row(0) is the position, head.row(1) is the velocity, head.row(2) is the acceleration, etc.
    Eigen::Matrix<double, 3, 2> head_nonuniform;
    head_nonuniform.row(0) << 0, 0;
    head_nonuniform.row(1) << 0, 0;
    head_nonuniform.row(2) << 0, 0;
    // the intermediate waypoints, each row is a waypoint.
    // because N = 4, there should be 3 waypoints.
    Eigen::Matrix<double, 3, 2> waypoints_nonuniform;
    waypoints_nonuniform.row(0) << 1, 0;
    waypoints_nonuniform.row(1) << 1, 1;
    waypoints_nonuniform.row(2) << 0, 1;
    // the end state. tail.row(0) is the position, tail.row(1) is the velocity, tail.row(2) is the acceleration, etc.
    Eigen::Matrix<double, 3, 2> tail_nonuniform;
    tail_nonuniform.row(0) << 0, 2;
    tail_nonuniform.row(1) << 0, 0;
    tail_nonuniform.row(2) << 0, 0;
    // time allocation
    Eigen::Vector<double, 4> Ts;
    Ts << 1, 2, 2, 1;
    // generate the trajectory
    minco_nonuniform.setConditions(head_nonuniform);
    minco_nonuniform.setParameters(waypoints_nonuniform, tail_nonuniform, Ts);
    // get the coefficients matrix of the trajectory
    cout << "NonUniform MINCO coefficients: " << endl << minco_nonuniform.getCoeffs() << endl;
    // get the energy of the trajectory
    cout << "NonUniform MINCO energy: " << minco_nonuniform.getEnergy() << endl;
    // get the energy partial derivatives by the coefficients
    cout << "NonUniform MINCO energy partial derivatives: " << endl << minco_nonuniform.getEnergyPartialGradByCoeffs() << endl;
    // get the energy partial derivatives by T
    cout << "NonUniform MINCO energy partial derivatives by T: " << minco_nonuniform.getEnergyPartialGradByTimes().transpose() << endl;
    // construct the trajectory
    PolynominalTrajectory<2> nonuniform_trajectory(minco_nonuniform.getCoeffs(), Ts);
    // get the position and velocity at time 2.5
    cout << "NonUniform MINCO position at time 2.5: " << nonuniform_trajectory.at(2.5, 0).transpose() << endl;
    cout << "NonUniform MINCO velocity at time 2.5: " << nonuniform_trajectory.at(2.5, 1).transpose() << endl;
    return 0;
}