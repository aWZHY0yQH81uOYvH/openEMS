#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include <cuda_runtime.h>

#include "engine.h"
#include "operator_cuda.h"

class Operator_CUDA;

class Engine_CUDA : public Engine
{
public:
	static Engine_CUDA* New(const Operator_CUDA* op, unsigned int cuda_device_number);

	virtual void Init();
	virtual void Reset();

	virtual void setCUDAdevice(unsigned int cuda_device_number);

	//!Iterate a number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	inline virtual FDTD_FLOAT GetVolt( unsigned int n, unsigned int x, unsigned int y, unsigned int z )		const { return ((FDTD_FLOAT*)&volt[x][y][z])[n]; }
	inline virtual FDTD_FLOAT GetVolt( unsigned int n, const unsigned int pos[3] )							const { return ((FDTD_FLOAT*)&volt[pos[0]][pos[1]][pos[2]])[n]; }
	inline virtual FDTD_FLOAT GetCurr( unsigned int n, unsigned int x, unsigned int y, unsigned int z )		const { return ((FDTD_FLOAT*)&curr[x][y][z])[n]; }
	inline virtual FDTD_FLOAT GetCurr( unsigned int n, const unsigned int pos[3] )							const { return ((FDTD_FLOAT*)&curr[pos[0]][pos[1]][pos[2]])[n]; }

	inline virtual void SetVolt( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)	{ ((FDTD_FLOAT*)&volt[x][y][z])[n]=value; }
	inline virtual void SetVolt( unsigned int n, const unsigned int pos[3], FDTD_FLOAT value )						{ ((FDTD_FLOAT*)&volt[pos[0]][pos[1]][pos[2]])[n]=value; }
	inline virtual void SetCurr( unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)	{ ((FDTD_FLOAT*)&curr[x][y][z])[n]=value; }
	inline virtual void SetCurr( unsigned int n, const unsigned int pos[3], FDTD_FLOAT value )						{ ((FDTD_FLOAT*)&curr[pos[0]][pos[1]][pos[2]])[n]=value; }

	Engine_CUDA(const Operator_CUDA* op);
	const Operator_CUDA* Op;

	unsigned int m_cuda_device_number;
	int m_supports_coop_launch;
	dim3 m_gridDim;
	dim3 m_blockDim;

	CUDA_VECTOR*** volt;
	CUDA_VECTOR*** curr;
};

#endif // ENGINE_CUDA_H
