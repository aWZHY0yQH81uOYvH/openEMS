#include "operator_cuda.h"
#include "engine_cuda.h"

#include "tools/array_ops.h"

#include <assert.h>

Operator_CUDA* Operator_CUDA::New(unsigned int cuda_device_number)
{
	cout << "Create FDTD operator (CUDA)" << endl;
	Operator_CUDA* op = new Operator_CUDA();
	op->setCUDAdevice(cuda_device_number);
	op->Init();
	return op;
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_CUDA::New(this, m_cuda_device_number);
	return m_Engine;
}

void Operator_CUDA::setCUDAdevice(unsigned int cuda_device_number) {
	m_cuda_device_number = cuda_device_number;
}

void Operator_CUDA::InitOperator()
{
	Delete3DArray_CUDA(vv,numLines);
	Delete3DArray_CUDA(vi,numLines);
	Delete3DArray_CUDA(iv,numLines);
	Delete3DArray_CUDA(ii,numLines);

	// Check that CUDA_VECTOR is the correct type
	assert(sizeof(CUDA_VECTOR) == sizeof(FDTD_FLOAT)*4);

	vv = Create3DArray_CUDA<CUDA_VECTOR>(numLines);
	vi = Create3DArray_CUDA<CUDA_VECTOR>(numLines);
	iv = Create3DArray_CUDA<CUDA_VECTOR>(numLines);
	ii = Create3DArray_CUDA<CUDA_VECTOR>(numLines);
}

void Operator_CUDA::Delete()
{
	CSX = NULL;

	Delete3DArray_CUDA(vv,numLines);
	Delete3DArray_CUDA(vi,numLines);
	Delete3DArray_CUDA(iv,numLines);
	Delete3DArray_CUDA(ii,numLines);
	vv=vi=iv=ii=0;
	delete MainOp; MainOp=0;
	for (int n=0; n<3; ++n)
	{
		delete[] EC_C[n];EC_C[n]=0;
		delete[] EC_G[n];EC_G[n]=0;
		delete[] EC_L[n];EC_L[n]=0;
		delete[] EC_R[n];EC_R[n]=0;
	}

	Delete_N_3DArray(m_epsR,numLines);
	m_epsR=0;
	Delete_N_3DArray(m_kappa,numLines);
	m_kappa=0;
	Delete_N_3DArray(m_mueR,numLines);
	m_mueR=0;
	Delete_N_3DArray(m_sigma,numLines);
	m_sigma=0;
}
