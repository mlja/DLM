// -*- c++ -*-
#pragma once

//! Normal row vector:
//! v: 1 2 3
template<typename T, std::size_t N>
class Vector
	: public Matrix<T, N, 1>
{
public:
	explicit Vector() noexcept
	{
		for(std::size_t i = 0u; i < N; ++i) // NOTE: Inline initialisation is broken in gcc 4.4
		{
			data[i] = 0;
		}
	}

	explicit Vector(double first, ...) noexcept
	{
		va_list ap;
		va_start(ap, first);
		data[0] = (T)first;
		for(std::size_t j = 1u; j < N; ++j)
		{
			data[j] = (T)va_arg(ap, double);
		}
		va_end(ap);
	}

	explicit Vector(const Matrix<T, N, 1>& other) noexcept
	{
		for(std::size_t j = 0u; j < N; ++j)
		{
			data[j] = other(j, 0);
		}
	}

	//! Accessor methods
	inline T& operator()(const std::size_t i) { return data[i]; }
	inline const T& operator()(const std::size_t i) const { return data[i]; }
	
	//! Create a new zero vector.
	static Vector zero() noexcept
	{
		Vector<T, N> out;
		return out;
	}

	//! Assign single column matrix to vector of same size
	Vector<T, N> operator=(const Matrix<T, N, 1>& other)
	{
		for(std::size_t i = 0u; i < N; ++i)
		{
			data[i] = other(i, 0);
		}
		return *this;
	}
};
