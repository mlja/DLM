// -*- c++ -*-
#pragma once

//! Constructors
template<typename T, std::size_t M, std::size_t N>
template<typename ...E>
constexpr Matrix<T, M, N>::Matrix(T first, E&&... elements) noexcept
	: data{first, std::forward<T>(static_cast<T>(elements))... }
{
	static_assert((sizeof...(elements) + 1) == N * M, "Number of elements doesn't match matrix size.");
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>::Matrix(const double values[]) noexcept
{
	for(std::size_t j = 0u; j < (M * N); ++j)
	{
		data[j] = (T)values[j];
	}
}

//! Cell accessor methods
template<typename T, std::size_t M, std::size_t N>
inline T& Matrix<T, M, N>::operator()(const std::size_t i, const std::size_t j)
{
	if( (i>M-1) || (j>N-1) )
	{ 
		throw std::out_of_range("Matrix: Index (i,j) out of range");
	}
	return data[i * N + j];
}

template<typename T, std::size_t M, std::size_t N>
inline const T& Matrix<T, M, N>::operator()(const std::size_t i, const std::size_t j) const
{
	if( (i>M-1) || (j>N-1) )
	{ 
		throw std::out_of_range("Matrix: Index (i,j) out of range");
	}
	return data[i * N + j];
}

template<typename T, std::size_t M, std::size_t N>
inline T& Matrix<T, M, N>::operator()(const std::size_t i)
{
	static_assert((M == 1) || (N == 1), "Only meaningful if M==1 or N==1, i.e. vectors");
	if(i >= ((M > N) ? M : N)) 
	{ 
		throw std::out_of_range("Matrix: Index (i) out of range");
	}
	return data[i];
}

template<typename T, std::size_t M, std::size_t N>
inline const T& Matrix<T, M, N>::operator()(const std::size_t i) const
{
	static_assert((M == 1) || (N == 1), "Only meaningful if M==1 or N==1, i.e. vectors");
	if(i >= ((M > N) ? M : N)) 
	{ 
		throw std::out_of_range("Matrix: Index (i) out of range"); 
	}
	return data[i];
}

//! Special matrices
// TODO: Create identity matrix using templating instead of a runtime loop.
template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::identity() noexcept
{
	Matrix<T, M, N> out;
	for(std::size_t i = 0u; i < ((M<N) ? M : N); ++i) // min(M,N)
	{
		out(i, i) = T_1;
	}
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::zero() noexcept
{
	Matrix<T, M, N> out; // filled with zeros by construction
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::ones() noexcept
{
	Matrix<T, M, N> out;
	out.data.fill(T_1);
	return out;
}

//! Scalar operations
template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator=(T scalar)
{
	data.fill(scalar);
	return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator+(T scalar)
{
	Matrix<T, M, N> out(*this);
	out += scalar;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator-(T scalar)
{
	Matrix<T, M, N> out(*this);
	out -= scalar;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator*(T scalar)
{
	Matrix<T, M, N> out(*this);
	out *= scalar;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator/(T scalar)
{
	Matrix<T, M, N> out(*this);
	out *= (T_1 / scalar);
	return out;
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator+=(T scalar)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] += scalar;
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator-=(T scalar)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] -= scalar;
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator*=(T scalar)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] *= scalar;
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator/=(T scalar)
{
	const T one_divided_by = T_1 / scalar;
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] *= one_divided_by;
	}
}

//!
//! Pointwise matrix operations
//!
template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator=(const Matrix<T, M, N>& other)
{
	data = other.data;
	return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator+(const Matrix<T, M, N>& other) const
{
	Matrix<T, M, N> out(*this);
	out += other;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator-(const Matrix<T, M, N>& other) const
{
	Matrix<T, M, N> out(*this);
	out -= other;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator*(const Matrix<T, M, N>& other) const
{
	Matrix<T, M, N> out(*this);
	out *= other;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator/(const Matrix<T, M, N>& other) const
{
	Matrix<T, M, N> out(*this);
	out /= other;
	return out;
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator+=(const Matrix<T, M, N>& other)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] += other.data[i];
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator-=(const Matrix<T, M, N>& other)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] -= other.data[i];
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator*=(const Matrix<T, M, N>& other)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] *= other.data[i];
	}
}

template<typename T, std::size_t M, std::size_t N>
void Matrix<T, M, N>::operator/=(const Matrix<T, M, N>& other)
{
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		data[i] /= other.data[i];
	}
}

//!
//! Matrix functionalities (multiplication, inversion, etc.)
//!
template<typename T, std::size_t M, std::size_t N>
template<std::size_t K>
Matrix<T, M, K> Matrix<T, M, N>::dot(const Matrix<T, N, K>& other) const
{
	Matrix<T, M, K> out; // zero initialized
	for(std::size_t i = 0u; i < M; ++i)
	{
		for(std::size_t k = 0u; k < N; ++k)
		{
			const T hold = data[i*N + k]; // *this(i, k)
			const T* p_other = &other(k, 0);
			T* p_out = &out(0, 0) + i*K; // out(i, 0)
			
			for(std::size_t j = 0u; j < K; ++j)
			{
				//out(i, j) += hold * other(k, j);
				(*p_out++) += hold * (*p_other++);
			}
		}
	}
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, N, M> Matrix<T, M, N>::t() const
{
	Matrix<T, N, M> out;
	const T* p_data = &data[0];
	
	for(std::size_t i = 0u; i < M; ++i)
	{
		T* p_out = &out(0, 0) + i; // out(0, i)
		for(std::size_t j = 0u; j < N; ++j)
		{
			//out(j, i) = (*this)(i, j);
			*p_out = (*p_data++);
			p_out += M; // step to next row (within column i)
		}
	}
	return out;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::inv_naive() const
{
	static_assert(N == M, "Only square matrices can be inverted.");
	Matrix<T, M, N> lhs = *this;
	Matrix<T, M, N> rhs = identity(); // Think of LHS and RHS as (N x 2N) array.

	for(std::size_t i = 0u; i < N; ++i) // Gaussian elimination, get reduced row echelon form on LHS
	{		
		assert(lhs(i, i) != T_0); // Grab diagonal entry and invert it (unless it is zero)
		const T scale = T_1 / lhs(i, i);
		for(std::size_t k = 0u; k < N; ++k) 
		{
			lhs(i, k) *= ((k>=i) ? scale : T_1); // starting at k=i on LHS
			rhs(i, k) *= scale; // starting at k=0 on RHS
		}

		for(std::size_t row = 0u; row < N; ++row) // LHS(i,i) is unity, now get zeros above and below
		{
			if(row == i) { continue; }
			const T temp = lhs(row, i); // an entry either vertically above or below (i,i)
			for(std::size_t k = 0u; k < N; ++k) // do elementary row operations
			{
				lhs(row, k) -= temp * lhs(i, k);
				rhs(row, k) -= temp * rhs(i, k);
			}
		}
	} // LHS is now the identity matrix, RHS is the matrix inverse
	return rhs;
}

template<typename T>
void inline swapEntries(T& a, T& b) {const T c = a; a = b; b = c;}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::inv() const
{
	static_assert(N == M, "Only square matrices can be inverted.");
	Matrix<T, M, N> lhs = *this;
	Matrix<T, M, N> rhs = identity();
	std::array<T, N> implicit {}; // container
	
	for(std::size_t i = 0u; i < N; ++i) // get implicit normalization for each row
	{	
		T abs_max = T_0; // reset
		for(std::size_t j = 0u; j < N; ++j)
		{	
			T temp = std::abs( lhs.data[i*N + j] );
			abs_max = (temp > abs_max) ? temp : abs_max;
		}
		if(abs_max == T_0) { throw("inv2(): Singular matrix (all zero row)"); }
		implicit[i] = T_1 / abs_max;
	}
	
	for(std::size_t j = 0u; j < N; ++j) // primary loop
	{
		T abs_max = T_0; // reset
		std::size_t row_pivot = 0u;
		for(std::size_t i = j; i < N; ++i) // identify pivot entry within column j, starting at i=j
		{
			T temp = implicit[i] * std::abs( lhs.data[i*N + j] );
			if(temp > abs_max)
			{
				abs_max = temp;
				row_pivot = i; // always >= j
			}
		}
		
		if(row_pivot != j) // if necessary, swap row j and the identified pivot row
		{
			swapEntries( implicit[row_pivot], implicit[j] ); // swap normalization constants
			for(std::size_t k = 0u; k < N; ++k) 
			{
				swapEntries( lhs.data[j*N + k], lhs.data[row_pivot*N + k] ); // LHS
				swapEntries( rhs.data[j*N + k], rhs.data[row_pivot*N + k] ); // RHS
			}
		}
		
		if(lhs.data[j*N + j] == T_0) { throw("inv2(): Singular matrix (zero pivot encountered)"); }
		const T scale = T_1 / lhs.data[j*N + j];
		for(std::size_t k = j; k < N; ++k)  // LHS, starting at k=j
		{
			lhs.data[j*N + k] *= scale;
		}
		for(std::size_t k = 0u; k < N; ++k) // RHS, starting at k=0
		{
			rhs.data[j*N + k] *= scale;
		}
		
		for(std::size_t i = 0u; i < N; ++i) // lhs(j,j) is now unity, go get zeros above and below
		{
			if(i == j) { continue; }			
			const T temp = lhs.data[i*N + j]; // an entry vertically above or below (j, j)
			for(std::size_t k = 0u; k < N; ++k)
			{
				lhs.data[i*N + k] -= temp * lhs.data[j*N + k]; // LHS
				rhs.data[i*N + k] -= temp * rhs.data[j*N + k]; // RHS
			}
		}
	}	
	return rhs;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, M> Matrix<T, M, N>::symmetric_wrap(const Matrix<T, N, N>& other) const
{
	Matrix<T, M, M> C; // C = (A * B) * A.T(), A=this(MxN), B=other(NxN), C=out(MxM)
	Matrix<T, M, N> D; // D = A * B, intermediate result, exploit B symmetric
	
	// compute D
	T* ptr_D = &D(0, 0);
	for(std::size_t i = 0u; i < M; ++i)
	{
		const T* ptr_B = &other(0, 0); // reset pointer
		for(std::size_t p = 0; p < N; ++p)
		{
			const T* ptr_A = &data[i*N]; // reset pointer (afo i)
			for(std::size_t k = 0u; k < N; ++k)
			{
				//D(i, p) += A(i, k) * B(p, k); // notice the swap (k, p) --> (p, k)
				*ptr_D += (*ptr_A++) * (*ptr_B++);
			}
			ptr_D++;
		}
	}
	
	// compute lower triangle of C (including diagonal)
	T* ptr_C = &C(0, 0);
	for(std::size_t i = 0u; i < M; ++i)
	{	
		const T* ptr_A = &data[0]; // reset pointer
		for(std::size_t j = 0; j <= i; ++j) // up to and including i
		{
			ptr_D = &D(i, 0); // reset pointer (afo i)
			for(std::size_t p = 0u; p < N; ++p)
			{
				//C(i, j) += D(i, p) * A(j, p); // notice the swap A.t(p, j) --> A(j, p)				
				*ptr_C += (*ptr_D++) * (*ptr_A++);
			}
			ptr_C++;
		}
		ptr_C += M - (i+1); // go to same position in next row, and then shift back to first column		
	}
	
	// populate entries across upper triangle of C (by symmetry)
	for(std::size_t i = 0u; i < M; ++i)
	{
		for(std::size_t j = i+1; j < M; ++j) 
		{
			C(i, j) = C(j, i);
		}
	}
	
	return C;
}

template<typename T, std::size_t M, std::size_t N>
T Matrix<T, M, N>::norminf(void) const
{
	T sum_abs = T_0;
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		sum_abs += std::abs( data[i] );
	}
	return sum_abs;
}

//!
//! Comparison
//!
template<typename T, std::size_t M, std::size_t N>
bool Matrix<T, M, N>::operator==(const Matrix<T, M, N>& other) const
{
	constexpr T eps = (T)0.000001; // 1e-6 suitable for float and double
	bool status = true;
	for(std::size_t i = 0u; i < (N * M); ++i)
	{
		status &= ( std::abs(data[i] - other.data[i]) < eps ); // becomes false if violated
	}
	return status;
}
