#pragma once

#include <cassert>
#include <algorithm>

#include <Eigen/Eigen>

template<int m>
class PolynominalTrajectory
{
private:
    Eigen::Matrix<double, Eigen::Dynamic, m> c;
    Eigen::VectorXd T_sum;
public:
    PolynominalTrajectory() = default;
    PolynominalTrajectory(const Eigen::Matrix<double, Eigen::Dynamic, m>& c_, const Eigen::VectorXd& Ts)
    {
        set_param(c_, Ts);
    }

    inline void set_param(const Eigen::Matrix<double, Eigen::Dynamic, m>& c_, const Eigen::VectorXd& Ts)
    {
        c = c_;
        T_sum = Ts;
        for(int i = 1; i < Ts.size(); i++)
            T_sum[i] += T_sum[i - 1];
    }

    inline int get_sections() const
    {
        return T_sum.size();
    }

    inline const Eigen::Matrix<double, Eigen::Dynamic, m>& get_coeffs() const
    {
        return c;
    }

    inline Eigen::Matrix<double, Eigen::Dynamic, m> get_coeffs(const size_t idx) const
    {
        const int _2s = c.rows() / T_sum.size();

        return c.block(idx * _2s, 0, _2s, 3);
    }

    inline double get_duration(const size_t idx) const
    {
        assert(idx < T_sum.size());
        return idx == 0 ? T_sum(0) : T_sum(idx) - T_sum(idx - 1);
    }

    inline Eigen::Vector<double, m> locally_at(const size_t Ti, const double t, const int order = 0) const
    {
        assert(t >= 0);
        const int _2s = c.rows() / T_sum.size();
        assert(order < _2s);

        double coeff = 1;
        for(int i = order; i > 1; i--) coeff *= i;
        Eigen::Vector<double, m> res; res.setZero();
        for(int i = order; i < _2s; i++) {
            res += coeff * c.row(Ti * _2s + i).transpose();
            coeff *= i + 1;
            coeff /= i - order + 1;
            coeff *= t;
        }
        return res;
    }

    typename Eigen::Vector<double, m> at(const double t, const int order = 0) const
    {
        const size_t Ti = std::lower_bound(T_sum.begin(), T_sum.end() - 1, t) - T_sum.begin();

        return locally_at(Ti, Ti == 0 ? t : t - T_sum[Ti - 1], order);
    }

    double get_total_duration() const
    {
        return T_sum.template tail<1>()(0);
    }

};
