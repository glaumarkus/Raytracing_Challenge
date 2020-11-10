#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#ifndef checkCudaErrors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
#endif

template <typename T>
class CUDAVector {
public:
	using ValueType = T;
public:

	__host__ CUDAVector() {
		size_ = 0;
		ReAlloc(2);
	}

	__host__ ~CUDAVector() {
		checkCudaErrors(cudaFree(data_));
	}

	__host__ void push_back(const T& value) {
		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		data_[size_++] = value;
	};

	__host__ void push_back(T&& value) {
		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		data_[size_++] = std::move(value);
	}

	// emplace
	template<typename... Args>
	__host__ T& emplace_back(Args&&... args) {

		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		new(&data_[size_]) T(std::forward<Args>(args)...);
		return data_[++size_];
	}

	__host__ __device__ size_t size() const { return size_; }

	__host__ __device__ const T& operator[](size_t idx) const { return data_[idx]; }
	__host__ __device__ T& operator[](size_t idx) { return data_[idx]; }


	__host__ __device__ CUDAVector& operator =(const CUDAVector& other) {
		size_ = other.size_;
		capacity_ = other.capacity_;
		data_ = other.data_;
		return *this;
	}

	__host__ __device__ CUDAVector& operator =(const CUDAVector&& other) {
		size_ = other.size_;
		capacity_ = other.capacity_;
		data_ = other.data_;
		return *this;
	}
	__host__ void reserve(size_t S) { ReAlloc(S); };

private:

	T* data_ = nullptr;
	size_t size_ = 0;
	size_t capacity_ = 0;


	__host__ void ReAlloc(size_t newCapacity)
	{

		if (newCapacity == 0) newCapacity = 2;
		// allocate new shared memory
		T* newData;
		size_t newSize = sizeof(T) * newCapacity;
		checkCudaErrors(cudaMallocManaged((void**)&newData, newSize));

		// copy contents
		if (newCapacity < size_)
			size_ = newCapacity;

		for (size_t i = 0; i < size_; ++i)
			newData[i] = std::move(data_[i]);

		//for (size_t i = 0; i < size_; ++i) 
		//	data_[i] .~T();

		// free memory
		checkCudaErrors(cudaFree(data_));

		//set ptr
		data_ = newData;
		capacity_ = newCapacity;
	}

};


/*
* Device only Vector
*/

template <typename T>
class DeviceVector {
private:

	T* data_ = nullptr;
	size_t size_ = 0;
	size_t capacity_ = 0;

	__host__ __device__ void ReAlloc(size_t newCapacity)
	{
		if (newCapacity == 0) newCapacity = 2;

		T* newData = (T*) ::operator new(newCapacity * sizeof(T));
		//T* newData = new T[newCapacity * sizeof(T)];

		size_t size = size_;
		if (newCapacity < size_)
			size_ = newCapacity;

		for (size_t i = 0; i < size_; ++i) {
			newData[i] = data_[i];
		}

		for (size_t i = 0; i < size_; ++i) {
			data_[i] .~T();
		}

		//delete[] data_;
		::operator delete(data_, capacity_ * sizeof(T));
		data_ = newData;
		capacity_ = newCapacity;
	}


public:

	__host__ __device__ DeviceVector() {
		size_ = 0;
		ReAlloc(2);
	}

	__host__ __device__ ~DeviceVector() {
		//delete[] data_;
	}

	__host__ __device__ void push_back(const T& value) {
		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		data_[size_++] = value;
	};

	__host__ __device__ void push_back(T&& value) {
		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		data_[size_++] = std::move(value);
	}

	// emplace
	template<typename... Args>
	__host__ __device__ T& emplace_back(Args&&... args) {
		if (size_ >= capacity_) ReAlloc(capacity_ * 2);
		new(&data_[size_]) T(std::forward<Args>(args)...);
		return data_[size_++];
	}

	__host__ __device__ size_t size() const { return size_; }

	__host__ __device__ const T& operator[](size_t idx) const { return data_[idx]; }
	__host__ __device__ T& operator[](size_t idx) { return data_[idx]; }

	__host__ __device__ DeviceVector& operator =(const DeviceVector& other) {
		size_ = other.size_;
		capacity_ = other.capacity_;
		data_ = other.data_;
		return *this;
	}

	__host__ __device__ DeviceVector& operator =(const DeviceVector&& other) {
		size_ = other.size_;
		capacity_ = other.capacity_;
		data_ = other.data_;
		return *this;
	}
	__host__ __device__ void reserve(size_t S) { ReAlloc(S); };

};

#endif