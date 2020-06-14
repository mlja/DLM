// -*- c++ -*-
#pragma once

#include <limits> // std::numeric_limits<T>::quiet_NaN() 
#include "Matrix.h"

//! Class for applying a Dynamic Linear Model (DLM)
//! Key reference from literature is the 1997 book by West&Harrison [W&H]
//! Most class methods defaults to standard DLM behavior (linear and closed)
//! Override functionalities as needed for each new application

//! Templated arguments:
//! T: type, float or double
//! n: state dimension
//! r: measurement dimension
//! Tauxil: auxiliary data type, eg nested struct or array

//! The auxiliary input is optional (and not a native part of the [W&H]-framework)

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil = void>
class DLM
{
public:
	DLM();
	~DLM() = default;
	
	void iterateDLM(const Vector<T, Tr>& Yt, const Tauxil* Zt = nullptr); // Zt auxiliary data
	T getStateEntry(std::size_t i);
	T getCovarEntry(std::size_t i, std::size_t j);
	
	//! These methods should (in principle) be called inside every specialized constructor
	void initState(const Vector<T, Tn>& m);
	void initStateCovar(const Matrix<T, Tn, Tn>& C);
	void initTransitionMatrix(const Matrix<T, Tn, Tn>& G);
	void initDesignMatrix(const Matrix<T, Tr, Tn>& F_T);
	void initEvolutionCovar(const Matrix<T, Tn, Tn>& W);
	void initMeasurementCovar(const Matrix<T, Tr, Tr>& V);

protected:
	virtual Vector<T, Tn> mapState(const Vector<T, Tn>& m);
	virtual Matrix<T, Tn, Tn> jacobiState(const Vector<T, Tn>& m);

	//! Intervention methods [W&H, Sec. 11.2], first arg not prefixed with const
	virtual void applyIntervention(Vector<T, Tn>& a, const Tauxil* Zt);
	virtual void interveneR(Matrix<T, Tn, Tn>& R, const Tauxil* Zt);
	
	virtual Vector<T, Tr> mapMeasurement(const Vector<T, Tn>& a);
	virtual Matrix<T, Tr, Tn> jacobiMeasurement(const Vector<T, Tn>& a);

	virtual void updateW(Matrix<T, Tn, Tn>& W, const Tauxil* Zt);

	virtual void updateV(Matrix<T, Tr, Tr>& V, const Tauxil* Zt,
						 Vector<T, Tr>& residual); //! no const, deliberate

	virtual Matrix<T, Tn, Tn> evaluateR(const Matrix<T, Tn, Tn>& G,
										const Matrix<T, Tn, Tn>& C,
										const Matrix<T, Tn, Tn>& W);

	virtual Matrix<T, Tr, Tr> evaluateQ(const Matrix<T, Tr, Tn>& F_T,
										const Matrix<T, Tn, Tn>& R,
										const Matrix<T, Tr, Tr>& V);

	virtual Matrix<T, Tn, Tr> evaluateA(const Matrix<T, Tn, Tn>& R,
										const Matrix<T, Tr, Tn>& F_T,
										const Matrix<T, Tr, Tr>& Q,
										const Matrix<T, Tn, Tn>& C,
										const Matrix<T, Tr, Tr>& V);

	virtual Matrix<T, Tn, Tn> evaluateC(const Matrix<T, Tn, Tn>& R,
										const Matrix<T, Tn, Tr>& A,
										const Matrix<T, Tr, Tn>& F_T);

private:
	// declare DLM vectors
	Vector<T, Tn> m; // (n x 1) state
	Vector<T, Tn> a; // (n x 1) prediction
	Vector<T, Tr> f; // (r x 1) forecast

	// declare DLM matrices
	Matrix<T, Tn, Tn> C;   // (n x n), state covariance
	Matrix<T, Tn, Tn> G;   // (n x n), state transition
	Matrix<T, Tn, Tn> W;   // (n x n), evolution covariance
	Matrix<T, Tn, Tn> R;   // (n x n), predicition/intervened covariance
	Matrix<T, Tr, Tn> F_T; // (r x n), design matrix
	Matrix<T, Tr, Tr> V;   // (r x r), measurement covariance
	Matrix<T, Tr, Tr> Q;   // (r x r), forecast covariance
	Matrix<T, Tn, Tr> A;   // (n x r), adaptive matrix (Kalman gain)
};

// set dummy entries to ensure frozen output (if broken usage)
// to be overruled inside every specialized class (after inheritance)
template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
DLM<T, Tn, Tr, Tauxil>::DLM()
{
	m   = Vector<T, Tn>::zero();
	C   = Matrix<T, Tn, Tn>::zero();
	G   = Matrix<T, Tn, Tn>::zero();
	F_T = Matrix<T, Tr, Tn>::zero();
	W   = Matrix<T, Tn, Tn>::zero();
	V   = Matrix<T, Tr, Tr>::identity();
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initState(const Vector<T, Tn>& m)
{
	this->m = m;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initStateCovar(const Matrix<T, Tn, Tn>& C)
{
	this->C = C;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initTransitionMatrix(const Matrix<T, Tn, Tn>& G)
{
	this->G = G;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initDesignMatrix(const Matrix<T, Tr, Tn>& F_T)
{
	this->F_T = F_T;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initEvolutionCovar(const Matrix<T, Tn, Tn>& W)
{
	this->W = W;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::initMeasurementCovar(const Matrix<T, Tr, Tr>& V)
{
	this->V = V;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Vector<T, Tn> DLM<T, Tn, Tr, Tauxil>::mapState(const Vector<T, Tn>& m)
{
	return Vector<T, Tn>(G * m); // Default: fixed linear transition G
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tn, Tn> DLM<T, Tn, Tr, Tauxil>::jacobiState(const Vector<T, Tn>& m)
{
	return G; // Default: fixed linear transition G
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::applyIntervention(Vector<T, Tn>& a, //! no const
											   const Tauxil* Zt)
{
	// Default: No intervention, override if needed
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Vector<T, Tr> DLM<T, Tn, Tr, Tauxil>::mapMeasurement(const Vector<T, Tn>& a)
{
	return Vector<T, Tr>(F_T * m); // Default: fixed linear design matrix F_T
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tr, Tn> DLM<T, Tn, Tr, Tauxil>::jacobiMeasurement(const Vector<T, Tn>& a)
{
	return F_T; // Default: fixed linear design matrix F_T
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::updateW(Matrix<T, Tn, Tn>& W, const Tauxil* Zt)
{
	// Default: Closed DLM, override if needed
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::updateV(Matrix<T, Tr, Tr>& V, const Tauxil* Zt,
									 Vector<T, Tr>& residual) //! no const
{
	// Default: Closed DLM, override if needed
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tn, Tn> DLM<T, Tn, Tr, Tauxil>::evaluateR(const Matrix<T, Tn, Tn>& G,
													const Matrix<T, Tn, Tn>& C, 
													const Matrix<T, Tn, Tn>& W)
{
	return G * C * G.T() + W; // override if needed (or to save flops)
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::interveneR(Matrix<T, Tn, Tn>& R, const Tauxil* Zt)
{
	// Default: No intervention, override if needed
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tr, Tr> DLM<T, Tn, Tr, Tauxil>::evaluateQ(const Matrix<T, Tr, Tn>& F_T,
													const Matrix<T, Tn, Tn>& R,
													const Matrix<T, Tr, Tr>& V)
{
	return F_T * R * F_T.T() + V; // override if needed (or to save flops)
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tn, Tr> DLM<T, Tn, Tr, Tauxil>::evaluateA(
								const Matrix<T, Tn, Tn>& R,
								const Matrix<T, Tr, Tn>& F_T,
								const Matrix<T, Tr, Tr>& Q,
								const Matrix<T, Tn, Tn>& C, // optional use (if needed)
								const Matrix<T, Tr, Tr>& V) // optional use (if needed)
{
	return R * F_T.T() * Q.inv(); // override if needed (or to save flops)
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
Matrix<T, Tn, Tn> DLM<T, Tn, Tr, Tauxil>::evaluateC(
								const Matrix<T, Tn, Tn>& R,
								const Matrix<T, Tn, Tr>& A,
								const Matrix<T, Tr, Tn>& F_T)
{
	//return R - A * Q * A.T(); // by plain definition
	return R - (A * F_T * R);
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
void DLM<T, Tn, Tr, Tauxil>::iterateDLM(const Vector<T, Tr>& Yt,
										const Tauxil* Zt)
{
	a = mapState(m);                    // a = G * m;
	G = jacobiState(m);
	applyIntervention(a, Zt);           //! manipulate prediction at will (no const)
	updateW(W, Zt);
	R = evaluateR(G, C, W);             // R = G * C * G.T() + W;
	interveneR(R, Zt);                  //! manipulate R at will (no const)

	f = mapMeasurement(a);              // f = F_T * a;
	F_T = jacobiMeasurement(a);
	Vector<T, Tr> residual(Yt - f);     //! allow for designs with "dangerous" behavior
	updateV(V, Zt, residual);           //! manipulate residual at will (no const)
	Q = evaluateQ(F_T, R, V);           // Q = F_T * R * F + V;

	A = evaluateA(R, F_T, Q, C, V);     // A = R * F * Q.inv();
	m = a + A * residual;               //! residual may have been changed by updateV()
	C = evaluateC(R, A, F_T);           // C = R - A * F_T * R;
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
T DLM<T, Tn, Tr, Tauxil>::getStateEntry(std::size_t i)
{
	return (i < Tn) ? m(i) : std::numeric_limits<T>::quiet_NaN();
}

template<typename T, std::size_t Tn, std::size_t Tr, typename Tauxil>
T DLM<T, Tn, Tr, Tauxil>::getCovarEntry(std::size_t i, std::size_t j)
{
	return ((i < Tn) && (j < Tn)) ? C(i,j) : std::numeric_limits<T>::quiet_NaN();
}

