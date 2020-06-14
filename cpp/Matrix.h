// -*- c++ -*-
#pragma once

#include <array>
#include <cassert>

//! T: type, eg float, double or long double
//! M and N: matrix dimensions
template<typename T, std::size_t M, std::size_t N>
class Matrix
{
public:
	Matrix() = default; // creates a zero matrix

	//! C++14 variadic constructor with strong type checks
	template<typename ...E>
	explicit constexpr Matrix(T element, E&&... elements) noexcept;

	explicit Matrix(const double values[]) noexcept; // initialise from double array

	//! Matrix accessor methods, and const versions (for making lookups in const matrices)
	inline T& operator()(const std::size_t i, const std::size_t j);
	inline const T& operator()(const std::size_t i, const std::size_t j) const;
	
	//! Vector accessor methods (taking a single arg only, and static assert on M==1 or N==1)
	inline T& operator()(const std::size_t i);
	inline const T& operator()(const std::size_t i) const;

	//! Special matrices
	static Matrix<T, M, N> identity() noexcept;
	static Matrix<T, M, N> zero() noexcept;
	static Matrix<T, M, N> ones() noexcept;

	//! Scalar operations (forwarded to corresponding inplace routine)
	Matrix<T, M, N> operator=(T scalar);
	Matrix<T, M, N> operator+(T scalar);
	Matrix<T, M, N> operator-(T scalar);
	Matrix<T, M, N> operator*(T scalar);
	Matrix<T, M, N> operator/(T scalar);
	// Inplace counterparts
	void operator+=(T scalar);
	void operator-=(T scalar);
	void operator*=(T scalar);
	void operator/=(T scalar);
	
	//! Pointwise matrix operations (forwarded to corresponding inplace routine)
	Matrix<T, M, N> operator=(const Matrix<T, M, N>& other);
	Matrix<T, M, N> operator+(const Matrix<T, M, N>& other) const;
	Matrix<T, M, N> operator-(const Matrix<T, M, N>& other) const;
	Matrix<T, M, N> operator*(const Matrix<T, M, N>& other) const;
	Matrix<T, M, N> operator/(const Matrix<T, M, N>& other) const;	
	// Inplace counterparts
	void operator+=(const Matrix<T, M, N>& other);
	void operator-=(const Matrix<T, M, N>& other);
	void operator*=(const Matrix<T, M, N>& other);
	void operator/=(const Matrix<T, M, N>& other);

	//! Matrix functionalities (multiplication, inversion, etc.)
	template<std::size_t K>
	Matrix<T, M, K> dot(const Matrix<T, N, K>& other) const;
	Matrix<T, N, M> t() const;
	Matrix<T, M, N> inv_naive() const; // reference implementation, fails easily
	Matrix<T, M, N> inv() const;
	Matrix<T, M, M> symmetric_wrap(const Matrix<T, N, N>& other) const;
	T norminf(void) const;
	
	//! Comparison
	bool operator==(const Matrix<T, M, N>& other) const;

	//! Typed zero and one
	static constexpr T T_0 = static_cast<T>(0);
	static constexpr T T_1 = static_cast<T>(1);

protected:
	std::array<T, M * N> data {}; // initialized, filled with zeros
};

#include "Matrix_impl.h"
