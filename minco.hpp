#pragma once

#include <cassert>

#include "Eigen/Eigen"

namespace minco {

template<int i, int j>
struct perm
{
    static constexpr int val = perm<i, j - 1>::val * (i - j + 1);
};
template<int i>
struct perm<i, 0>
{
    static constexpr int val = 1;
};

template<int... _Ip>
    using int_sequence = std::integer_sequence<int, _Ip...>;
template<int _Np>
    using make_int_sequence = std::make_integer_sequence<int, _Np>;

template<int lowerBw, int upperBw>
class BandedMatrix
{
    static_assert(lowerBw >= 0 && upperBw >= 0);
public:
    // The size of A, as well as the lower/upper
    // banded width p/q are needed
    inline void create(const int &n)
    {
        mat.resize(lowerBw + upperBw + 1, n);
        mat.setZero();
    }

private:
    Eigen::Matrix<double, lowerBw + upperBw + 1, Eigen::Dynamic> mat;

public:
    // Reset the matrix to zero
    inline void reset(void)
    {
        mat.setZero();
    }

    // The band matrix is stored as suggested in "Matrix Computation"
    inline const double &operator()(const int &i, const int &j) const
    {
        return mat(i - j + upperBw, j);
    }

    inline double &operator()(const int &i, const int &j)
    {
        return mat(i - j + upperBw, j);
    }

    // This function conducts banded LU factorization in place
    // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
    inline void factorizeLU()
    {
        double cVl;
        for (int k = 0; k <= mat.cols() - 2; k++)
        {
            const int iM = std::min(k + lowerBw, int(mat.cols() - 1));
            cVl = operator()(k, k);
            for (int i = k + 1; i <= iM; i++)
                if (operator()(i, k) != 0.0)
                    operator()(i, k) /= cVl;
            const int jM = std::min(k + upperBw, int(mat.cols() - 1));
            for (int j = k + 1; j <= jM; j++)
            {
                cVl = operator()(k, j);
                if (cVl != 0.0)
                    for (int i = k + 1; i <= iM; i++)
                        if (operator()(i, k) != 0.0)
                            operator()(i, j) -= operator()(i, k) * cVl;
            }
        }
    }

    // This function solves Ax=b, then stores x in b
    // The input b is required to be N*m, i.e.,
    // m vectors to be solved.
    template <typename EIGENMAT>
    inline void solve(EIGENMAT &b) const
    {
        for (int j = 0; j <= mat.cols() - 1; j++)
        {
            const int iM = std::min(j + lowerBw, int(mat.cols() - 1));
            for (int i = j + 1; i <= iM; i++)
                if (operator()(i, j) != 0.0)
                    b.row(i) -= operator()(i, j) * b.row(j);
        }
        for (int j = mat.cols() - 1; j >= 0; j--)
        {
            b.row(j) /= operator()(j, j);
            const int iM = std::max(0, j - upperBw);
            for (int i = iM; i <= j - 1; i++)
                if (operator()(i, j) != 0.0)
                    b.row(i) -= operator()(i, j) * b.row(j);
        }
    }

    // This function solves ATx=b, then stores x in b
    // The input b is required to be N*m, i.e.,
    // m vectors to be solved.
    template <typename EIGENMAT>
    inline void solveAdj(EIGENMAT &b) const
    {
        for (int j = 0; j <= mat.cols() - 1; j++)
        {
            b.row(j) /= operator()(j, j);
            const int iM = std::min(j + upperBw, int(mat.cols() - 1));
            for (int i = j + 1; i <= iM; i++)
                if (operator()(j, i) != 0.0)
                    b.row(i) -= operator()(j, i) * b.row(j);
        }
        for (int j = mat.cols() - 1; j >= 0; j--)
        {
            const int iM = std::max(0, j - lowerBw);
            for (int i = iM; i <= j - 1; i++)
                if (operator()(j, i) != 0.0)
                    b.row(i) -= operator()(j, i) * b.row(j);
        }
    }
};

constexpr int NonUniform = 0;
constexpr int Uniform = 1;

template<int s, int option>
struct MINCO_T_op;
template<int s>
struct MINCO_T_op<s, NonUniform>
{
    typedef Eigen::Matrix<double, 2 * s, Eigen::Dynamic> T_Type;
    template<int i>
    static inline double get_pow_T(const T_Type &T, const int Ti) { return T(i, Ti); }
    template<int i>
    static inline double get_pow_norm_T(const T_Type &T, const int Ti) { return T(i, Ti); }
};
template<int s>
struct MINCO_T_op<s, Uniform>
{
    typedef Eigen::Vector<double, 2 * s> T_Type;
    template<int i>
    static inline double get_pow_T(const T_Type &T, const int Ti) { return T(i); }
    template<int i>
    static inline constexpr double get_pow_norm_T(const T_Type &T, const int Ti) { return 1.0; }
};

template<int s, int m, int option>
struct MINCO_op
{
    typedef BandedMatrix<2 * s, 2 * s> A_Type;
    typedef Eigen::Matrix<double, Eigen::Dynamic, m> C_Type;
    typedef typename MINCO_T_op<s, option>::T_Type T_Type;
    typedef MINCO_T_op<s, option> T_op;

    template<int... i>
    static inline void fillF0(A_Type &A, int_sequence<i...>)
    {
        ((
            A(i, i) = perm<i, i>::val
        ), ...);
    }

    template<int j, int... k>
    static inline void fillEi_final(A_Type &A, const T_Type &T, const int i, int_sequence<k...>)
    {
        ((
            A((2 * i + 1) * s + (j + s) % (2 * s), 2 * i * s + j + k) = perm<j + k, j>::val * T_op::template get_pow_norm_T<k>(T, i)
        ), ...);
    }
    template<int... j>
    static inline void fillEi_rest(A_Type &A, const T_Type &T, const int i, int_sequence<j...>)
    {
        (
            fillEi_final<j>(A, T, i, make_int_sequence<2 * s - j>())
        , ...);
    }
    template<int... j>
    static inline void fillEi(A_Type &A, const T_Type &T, const int i, int_sequence<j...>)
    {
        ((
            A((2 * i + 2) * s - 1, 2 * i * s + j) = T_op::template get_pow_norm_T<j>(T, i)
        ), ...);
        fillEi_rest(A, T, i, make_int_sequence<2 * s - 1>());
    }

    template<int... j>
    static inline void fillFi(A_Type &A, const int i, int_sequence<j...>)
    {
        ((
            A((2 * i + 1) * s + (j + s) % (2 * s), (2 * i + 2) * s + j) = -perm<j, j>::val
        ), ...);
    }

    template<int i, int... j>
    static inline void fillEN_final(A_Type &A, const T_Type &T, const int N, int_sequence<j...>)
    {
        ((
            A((2 * N - 1) * s + i, (2 * N - 2) * s + i + j) = perm<i + j, i>::val * T_op::template get_pow_norm_T<j>(T, N - 1)
        ), ...);
    }
    template<int... i>
    static inline void fillEN(A_Type &A, const T_Type &T, const int N, int_sequence<i...>)
    {
        (
            fillEN_final<i>(A, T, N, make_int_sequence<2 * s - i>())
        , ...);
    }

    template<int i, int... j>
    static inline double getEnergy_edge_final(const C_Type &c, const T_Type &T, const int Ti, int_sequence<j...>)
    {
        return ((
            (double)perm<s + i, s>::val * perm<s + i + j + 1, s>::val / (2 * i + j + 2) * c.row((2 * Ti + 1) * s + i).dot(c.row((2 * Ti + 1) * s + i + j + 1)) * T_op::template get_pow_T<2 * i + j + 2>(T, Ti)
        ) + ... + 0);
    }
    template<int... i>
    static inline double getEnergy_edge(const C_Type &c, const T_Type &T, const int Ti, int_sequence<i...>)
    {
        return (
            getEnergy_edge_final<i>(c, T, Ti, make_int_sequence<s - i - 1>())
        + ...);
    }
    template<int... i>
    static inline double getEnergy_diag(const C_Type &c, const T_Type &T, const int Ti, int_sequence<i...>)
    {
        return ((
            (double)perm<s + i, s>::val * perm<s + i, s>::val / (2 * i + 1) * c.row((2 * Ti + 1) * s + i).squaredNorm() * T_op::template get_pow_T<2 * i + 1>(T, Ti)
        ) + ...);
    }

    template<int i, int... j>
    static inline void getEnergyPartialGradByCoeffs_line(const C_Type &c, const T_Type &T, const int Ti, C_Type &gdC, int_sequence<j...>)
    {
        gdC.row((2 * Ti + 1) * s + i) = 2 * ((
            (double)perm<s + i, s>::val * perm<s + j, s>::val / (i + j + 1) * c.row((2 * Ti + 1) * s + j) * T_op::template get_pow_T<i + j + 1>(T, Ti)
        ) + ...);
    }
    template<int... i>
    static inline void getEnergyPartialGradByCoeffs(const C_Type &c, const T_Type &T, const int Ti, C_Type &gdC, int_sequence<i...>)
    {
        ((
            getEnergyPartialGradByCoeffs_line<i>(c, T, Ti, gdC, make_int_sequence<s>())
        ), ...);
    }

    template<int i, int... j>
    static inline double getEnergyPartialGradByTimes_edge_final(const C_Type &c, const T_Type &T, const int Ti, int_sequence<j...>)
    {
        return ((
            (double)perm<s + i, s>::val * perm<s + i + j + 1, s>::val * c.row((2 * Ti + 1) * s + i).dot(c.row((2 * Ti + 1) * s + i + j + 1)) * T_op::template get_pow_T<2 * i + j + 1>(T, Ti)
        ) + ... + 0);
    }
    template<int... i>
    static inline double getEnergyPartialGradByTimes_edge(const C_Type &c, const T_Type &T, const int Ti, int_sequence<i...>)
    {
        return (
            getEnergyPartialGradByTimes_edge_final<i>(c, T, Ti, make_int_sequence<s - i - 1>())
        + ...);
    }
    template<int... i>
    static inline double getEnergyPartialGradByTimes_diag(const C_Type &c, const T_Type &T, const int Ti, int_sequence<i...>)
    {
        return ((
            (double)perm<s + i, s>::val * perm<s + i, s>::val * c.row((2 * Ti + 1) * s + i).squaredNorm() * T_op::template get_pow_T<2 * i>(T, Ti)
        ) + ...);
    }

    template<int order, int i, int Ns, int... j>
    static inline void at_T_end_mat(const C_Type &c, const T_Type &T, const int Ti, Eigen::Matrix<double, Ns, m> &B, int_sequence<j...>)
    {
        // get the specific order point at index Ti of T, and put it in B.row(i) 
        // usage: at_T_end_mat<order, i>(Ti, B, make_int_sequence<2 * s - order>())
        B.row(i) = ((
            (double)perm<order + j, order>::val * T_op::template get_pow_T<j>(T, Ti) * c.row(2 * Ti * s + order + j)
        ) + ...);
    }
};


template<int s, int m, int option = NonUniform>
class MINCO;

template<int s, int m>
class MINCO<s, m, NonUniform>
{
    static_assert(s >= 1 && m >= 1);
    typedef MINCO_op<s, m, NonUniform> Op;
private:
    const int N;
    Eigen::Matrix<double, s, m> head;
    Eigen::Matrix<double, s, m> tail;
    BandedMatrix<2 * s, 2 * s> A;
    Eigen::Matrix<double, Eigen::Dynamic, m> c;
    Eigen::Matrix<double, 2 * s, Eigen::Dynamic> T;

public:
    MINCO(const int pieceNum): N(pieceNum)
    {
        assert(pieceNum >= 1);
        
        A.create(2 * s * N);
        c.resize(2 * s * N, m);
        T.resize(2 * s, N);
        T.row(0).setOnes();
    }

    inline void setConditions(const Eigen::Matrix<double, s, m> &headState)
    {
        head = headState;
    }

    inline void setParameters(const Eigen::Matrix<double, Eigen::Dynamic, m> &inPs,
                              const Eigen::Matrix<double, s, m> &tailState,
                              const Eigen::VectorXd &ts)
    {
        assert(ts.size() == N && inPs.rows() == N - 1);

        tail = tailState;

        for(int i = 1; i < 2 * s; i++)
            T.row(i) = T.row(i - 1).cwiseProduct(ts.transpose());

        A.reset();

        c.setZero();
        c.template topRows<s>() = head;
        c.template bottomRows<s>() = tail;

        Op::fillF0(A, make_int_sequence<s>());
        for(int i = 0; i < N - 1; i++) {
            Op::fillEi(A, T, i, make_int_sequence<2 * s>());
            Op::fillFi(A, i, make_int_sequence<2 * s - 1>());
            c.row((2 * i + 2) * s - 1) = inPs.row(i);
        }
        Op::fillEN(A, T, N, make_int_sequence<s>());

        A.factorizeLU();
        A.solve(c);
    }

    inline const Eigen::Matrix<double, Eigen::Dynamic, m> &getCoeffs() const
    {
        return c;
    }

    inline double getEnergy() const
    {
        double energy = 0;
        for(int Ti = 0; Ti < N; Ti++)
            energy += Op::getEnergy_diag(c, T, Ti, make_int_sequence<s>()) + 
                  2 * Op::getEnergy_edge(c, T, Ti, make_int_sequence<s>());
        return energy;
    }

    inline Eigen::Matrix<double, Eigen::Dynamic, m> getEnergyPartialGradByCoeffs() const
    {
        Eigen::Matrix<double, Eigen::Dynamic, m> gdC(2 * s * N, m);
        for(int Ti = 0; Ti < N; Ti++) {
            gdC.template block<s, m>(2 * Ti * s, 0).setZero();
            Op::getEnergyPartialGradByCoeffs(c, T, Ti, gdC, make_int_sequence<s>());
        }
        return gdC;
    }

    inline Eigen::VectorXd getEnergyPartialGradByTimes() const
    {
        Eigen::VectorXd gdT(N);
        for(int Ti = 0; Ti < N; Ti++)
            gdT[Ti] = Op::getEnergyPartialGradByTimes_diag(c, T, Ti, make_int_sequence<s>()) + 
                  2 * Op::getEnergyPartialGradByTimes_edge(c, T, Ti, make_int_sequence<s>());
        return gdT;
    }

    inline void attachGrad(const Eigen::Matrix<double, Eigen::Dynamic, m> &partialGradByCoeffs,
                           Eigen::Matrix<double, Eigen::Dynamic, m> &gradByPoints,
                           Eigen::Matrix<double, s, m> &gradByTail,
                           Eigen::VectorXd &gradByTimes) const
    {
        assert(partialGradByCoeffs.rows() == 2 * N * s && gradByPoints.rows() == N - 1 && gradByTimes.size() == N);

        Eigen::Matrix<double, Eigen::Dynamic, m> adjGrad = partialGradByCoeffs;
        A.solveAdj(adjGrad);

        for (int i = 0; i < N - 1; i++)
            gradByPoints.row(i) += adjGrad.row(2 * s * i + 2 * s - 1);
        
        gradByTail += adjGrad.template bottomRows<s>();

        Eigen::Matrix<double, 2 * s, m> B1;
        for(int Ti = 0; Ti < N - 1; Ti++) {
            fillB1(Ti, B1, make_int_sequence<2 * s - 2>());
            gradByTimes(Ti) -= B1.cwiseProduct(adjGrad.template block<2 * s, m>(2 * s * Ti + s, 0)).sum();
        }
        Eigen::Matrix<double, s, m> B2;
        B2.template topRows<s - 1>() = tail.template bottomRows<s - 1>();
        Op::template at_T_end_mat<s, s - 1>(c, T, N - 1, B2, make_int_sequence<s>());
        gradByTimes(N - 1) -= B2.cwiseProduct(adjGrad.template bottomRows<s>()).sum();
    }

private:
    template<int... i>
    inline void fillB1(const int Ti, Eigen::Matrix<double, 2 * s, m> &B1, int_sequence<i...>) const
    {
        // usage: fillB1(Ti, B1, make_int_sequence<2 * s - 2>())
        Op::template at_T_end_mat<1, s - 1>(c, T, Ti, B1, make_int_sequence<2 * s - 1>());
        B1.row(s) = B1.row(s - 1);
        (
            Op::template at_T_end_mat<i + 2, (i + s + 1) % (2 * s)>(c, T, Ti, B1, make_int_sequence<2 * s - i - 2>())
        , ...);
    }
};

template<int s, int m>
class MINCO<s, m, Uniform>
{
    static_assert(s >= 1 && m >= 1);
    typedef MINCO_op<s, m, Uniform> Op;
private:
    const int N;
    Eigen::Matrix<double, s, m> head;
    Eigen::Matrix<double, s, m> tail;
    BandedMatrix<2 * s, 2 * s> A;
    Eigen::Matrix<double, Eigen::Dynamic, m> b, c;
    Eigen::Vector<double, 2 * s> T, Tinv;

public:
    MINCO(const int pieceNum): N(pieceNum)
    {
        assert(pieceNum >= 1);
        
        A.create(2 * s * N);
        b.resize(2 * s * N, m);
        c.resize(2 * s * N, m);
        T(0) = 1.0;

        Op::fillF0(A, make_int_sequence<s>());
        for(int i = 0; i < N - 1; i++) {
            Op::fillEi(A, T, i, make_int_sequence<2 * s>());
            Op::fillFi(A, i, make_int_sequence<2 * s - 1>());
        }
        Op::fillEN(A, T, N, make_int_sequence<s>());

        A.factorizeLU();
    }

    inline void setConditions(const Eigen::Matrix<double, s, m> &headState)
    {
        head = headState;
    }

    inline void setParameters(const Eigen::Matrix<double, Eigen::Dynamic, m> &inPs,
                              const Eigen::Matrix<double, s, m> &tailState,
                              const double dT)
    {
        assert(inPs.rows() == N - 1);

        tail = tailState;

        for(int i = 1; i < 2 * s; i++)
            T(i) = T(i - 1) * dT;
        Tinv = T.cwiseInverse();

        b.setZero();
        b.template topRows<s>() = head.array().colwise() * T.template head<s>().array();
        b.template bottomRows<s>() = tail.array().colwise() * T.template head<s>().array();
        for(int i = 0; i < N - 1; i++)
            b.row((2 * i + 2) * s - 1) = inPs.row(i);

        A.solve(b);

        for (int i = 0; i < N; i++)
            c.template block<2 * s, m>(2 * s * i, 0) = b.template block<2 * s, m>(2 * s * i, 0).array().colwise() * Tinv.array();
    }

    inline const Eigen::Matrix<double, Eigen::Dynamic, m> &getCoeffs() const
    {
        return c;
    }

    inline double getEnergy() const
    {
        double energy = 0;
        for(int Ti = 0; Ti < N; Ti++)
            energy += Op::getEnergy_diag(c, T, Ti, make_int_sequence<s>()) + 
                  2 * Op::getEnergy_edge(c, T, Ti, make_int_sequence<s>());
        return energy;
    }

    inline Eigen::Matrix<double, Eigen::Dynamic, m> getEnergyPartialGradByCoeffs() const
    {
        Eigen::Matrix<double, Eigen::Dynamic, m> gdC(2 * s * N, m);
        for(int Ti = 0; Ti < N; Ti++) {
            gdC.template block<s, m>(2 * Ti * s, 0).setZero();
            Op::getEnergyPartialGradByCoeffs(c, T, Ti, gdC, make_int_sequence<s>());
        }
        return gdC;
    }

    inline double getEnergyPartialGradBydT() const
    {
        double gdT = 0;
        for(int Ti = 0; Ti < N; Ti++)
            gdT += Op::getEnergyPartialGradByTimes_diag(c, T, Ti, make_int_sequence<s>()) + 
               2 * Op::getEnergyPartialGradByTimes_edge(c, T, Ti, make_int_sequence<s>());
        return gdT;
    }

    inline void attachGrad(const Eigen::Matrix<double, Eigen::Dynamic, m> &partialGradByCoeffs,
                           Eigen::Matrix<double, Eigen::Dynamic, m> &gradByPoints,
                           Eigen::Matrix<double, s, m> &gradByTail,
                           double &gradBydT) const
    {
        assert(partialGradByCoeffs.rows() == 2 * N * s && gradByPoints.rows() == N - 1);

        Eigen::Matrix<double, Eigen::Dynamic, m> adjScaledGrad(2 * N * s, m);
        for (int i = 0; i < N; i++)
            adjScaledGrad.template block<2 * s, m>(2 * s * i, 0) = partialGradByCoeffs.template block<2 * s, m>(2 * s * i, 0).array().colwise() * Tinv.array();
        A.solveAdj(adjScaledGrad);

        for (int i = 0; i < N - 1; i++)
            gradByPoints.row(i) += adjScaledGrad.row(2 * s * i + 2 * s - 1);
        
        gradByTail += (adjScaledGrad.template bottomRows<s>().array().colwise() * T.template head<s>().array()).matrix();
        
        attachEdgeToGdT(gradBydT, adjScaledGrad, make_int_sequence<s - 1>());
        Eigen::Vector<double, 2 * s> gdTinv; getGdTinv(gdTinv, make_int_sequence<2 * s - 1>());
        const Eigen::VectorXd &gdcol = partialGradByCoeffs.cwiseProduct(b).rowwise().sum();
        for(int i = 0; i < N; i++)
            gradBydT += gdTinv.dot(gdcol.segment<2 * s>(2 * s * i));
    }

private:
    template<int... i>
    inline void attachEdgeToGdT(double &gdT, const Eigen::Matrix<double, Eigen::Dynamic, m> &adjScaledGrad, int_sequence<i...>) const
    {
        gdT += ((
            head.row(i + 1).dot(adjScaledGrad.row(i + 1)) * (i + 1) * T(i)
        ) + ...);
        gdT += ((
            tail.row(i + 1).dot(adjScaledGrad.template bottomRows<s>().row(i + 1)) * (i + 1) * T(i)
        ) + ...);
    }

    template<int ...i>
    inline void getGdTinv(Eigen::Vector<double, 2 * s> &gdTinv, int_sequence<i...>) const
    {
        ((
            gdTinv(i) = -i * Tinv(i + 1)
        ), ...);
        gdTinv(2 * s - 1) = -(2 * s - 1) * Tinv(2 * s - 1) * Tinv(1);
    }
};

}