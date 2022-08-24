#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

#include "engine_cuda.h"
#include "operator.h"

//! CUDA FDTD-operator
class Operator_CUDA : public Operator
{
	friend class Engine_CUDA;

public:
	static Operator_CUDA* New(unsigned int cuda_device_number = 0);

	virtual void setCUDAdevice(unsigned int cuda_device_number);

	void InitOperator();
	void Delete();

	virtual Engine* CreateEngine();

	inline virtual FDTD_FLOAT GetVV( unsigned int n, unsigned int x, unsigned int y, unsigned int z ) const { return ((FDTD_FLOAT*)&vv[x][y][z])[n]; }
	inline virtual FDTD_FLOAT GetVI( unsigned int n, unsigned int x, unsigned int y, unsigned int z ) const { return ((FDTD_FLOAT*)&vi[x][y][z])[n]; }
	inline virtual FDTD_FLOAT GetII( unsigned int n, unsigned int x, unsigned int y, unsigned int z ) const { return ((FDTD_FLOAT*)&ii[x][y][z])[n]; }
	inline virtual FDTD_FLOAT GetIV( unsigned int n, unsigned int x, unsigned int y, unsigned int z ) const { return ((FDTD_FLOAT*)&iv[x][y][z])[n]; }

	inline virtual void SetVV( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value ) { ((FDTD_FLOAT*)&vv[x][y][z])[n] = value; }
	inline virtual void SetVI( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value ) { ((FDTD_FLOAT*)&vi[x][y][z])[n] = value; }
	inline virtual void SetII( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value ) { ((FDTD_FLOAT*)&ii[x][y][z])[n] = value; }
	inline virtual void SetIV( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value ) { ((FDTD_FLOAT*)&iv[x][y][z])[n] = value; }

	//EC operator
	CUDA_VECTOR*** vv; //calc new voltage from old voltage
	CUDA_VECTOR*** vi; //calc new voltage from old current
	CUDA_VECTOR*** ii; //calc new current from old current
	CUDA_VECTOR*** iv; //calc new current from old voltage

protected:
	unsigned int m_cuda_device_number;
};

#endif // OPERATOR_CUDA_H
