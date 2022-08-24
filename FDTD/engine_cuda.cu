#include "engine_cuda.h"

#include "tools/array_ops.h"

#include <cooperative_groups.h>

//! \brief construct an Engine instance
//! it's the responsibility of the caller to free the returned pointer
Engine_CUDA* Engine_CUDA::New(const Operator_CUDA* op, unsigned int cuda_device_number)
{
	cout << "Create FDTD engine (CUDA)" << endl;
	Engine_CUDA* e = new Engine_CUDA(op);
	e->setCUDAdevice(cuda_device_number);
	e->Init();
	return e;
}

Engine_CUDA::Engine_CUDA(const Operator_CUDA* op) : Engine::Engine(op)
{
	m_cuda_device_number = 0;
	m_supports_coop_launch = 0;
	m_gridDim = {0};
	m_blockDim = {0};
	Op = op;
	m_type = CUDA;
}

void Engine_CUDA::setCUDAdevice(unsigned int cuda_device_number) {
	m_cuda_device_number = cuda_device_number;
}

void Engine_CUDA::Init()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
  	if(nDevices <= 0)
		throw std::runtime_error("No CUDA devices found");
	if(m_cuda_device_number >= nDevices)
		throw std::runtime_error("CUDA device number out of range");
	m_cuda_device_number = m_cuda_device_number;
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_cuda_device_number);
    cout << "  Running on device: " << prop.name << endl;

	cudaSetDevice(m_cuda_device_number);
	cudaDeviceGetAttribute(&m_supports_coop_launch, cudaDevAttrCooperativeLaunch, m_cuda_device_number);

	m_blockDim = {10, 10, 10};
	m_gridDim = {
		(unsigned int)ceilf(numLines[0]/m_blockDim.x),
		(unsigned int)ceilf(numLines[1]/m_blockDim.y),
		(unsigned int)ceilf(numLines[2]/m_blockDim.z)
	};

	cout << "numLines: " << numLines[0] << ", " << numLines[1] << ", " << numLines[2] << endl;
	cout << "grid dim: " << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << endl;

	numTS = 0;
	volt = Create3DArray_CUDA<CUDA_VECTOR>(numLines);
	curr = Create3DArray_CUDA<CUDA_VECTOR>(numLines);

	InitExtensions();
	SortExtensionByPriority();
}

void Engine_CUDA::Reset()
{
	Delete3DArray_CUDA(volt,numLines);
	volt=NULL;
	Delete3DArray_CUDA(curr,numLines);
	curr=NULL;

	ClearExtensions();
}

__device__ void UpdateVoltages(CUDA_VECTOR ***volt, CUDA_VECTOR ***curr, CUDA_VECTOR ***opvi, CUDA_VECTOR ***opvv, unsigned int x, unsigned int y, unsigned int z)
{
	CUDA_VECTOR v = volt[x][y][z];
	CUDA_VECTOR i = curr[x][y][z];
	CUDA_VECTOR ix = curr[x-(x!=0)][y][z];
	CUDA_VECTOR iy = curr[x][y-(y!=0)][z];
	CUDA_VECTOR iz = curr[x][y][z-(z!=0)];
	CUDA_VECTOR vi = opvi[x][y][z];
	CUDA_VECTOR vv = opvv[x][y][z];

	v.x = v.x * vv.x + vi.x * (i.z - iy.z - i.y + iz.y);
	v.y = v.y * vv.y + vi.y * (i.x - iz.x - i.z + ix.z);
	v.z = v.y * vv.y + vi.z * (i.y - ix.y - i.x + iy.x);

	volt[x][y][z] = v;
}

__device__ void UpdateCurrents(CUDA_VECTOR ***volt, CUDA_VECTOR ***curr, CUDA_VECTOR ***opiv, CUDA_VECTOR ***opii, unsigned int x, unsigned int y, unsigned int z)
{
	CUDA_VECTOR i = curr[x][y][z];
	CUDA_VECTOR v = volt[x][y][z];
	CUDA_VECTOR vx = volt[x+1][y][z];
	CUDA_VECTOR vy = volt[x][y+1][z];
	CUDA_VECTOR vz = volt[z][y][z+1];
	CUDA_VECTOR iv = opiv[x][y][z];
	CUDA_VECTOR ii = opii[x][y][z];

	i.x = i.x * ii.x + iv.x * (v.z - vy.z - v.y + vz.y);
	i.y = i.y * ii.y + iv.y * (v.x - vz.x - v.z + vx.z);
	i.z = i.z * ii.z + iv.z * (v.y - vx.y - v.x + vy.x);

	curr[x][y][z] = i;
}

/*__global__ void ManyTS(Engine_CUDA *instance, unsigned int startTS, unsigned int iterTS, unsigned int numLines[])
{
	const unsigned int pos[] = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		blockDim.z * blockIdx.z + threadIdx.z
	};

	if(pos[0] >= numLines[0] || pos[1] >= numLines[1] || pos[2] >= numLines[2]) return;

	const unsigned int endTS = startTS + iterTS;
	bool runUpdateCurrents = (pos[0] < numLines[0]-1 && pos[1] < numLines[1]-1 && pos[2] < numLines[2]-1);

	cooperative_groups::grid_group grid = cooperative_groups::this_grid();

	for (unsigned int ts=startTS; ts<endTS; ++ts)
	{
		//voltage updates with extensions
		//DoPreVoltageUpdates();
		UpdateVoltages(instance, pos[0], pos[1], pos[2]);
		//DoPostVoltageUpdates();
		//Apply2Voltages();

		grid.sync();

		//current updates with extensions
		//DoPreCurrentUpdates();
		if(runUpdateCurrents) UpdateCurrents(instance, pos[0], pos[1], pos[2]);
		//DoPostCurrentUpdates();
		//Apply2Current();

		grid.sync();
	}
}*/

__global__ void VoltageKernel(CUDA_VECTOR ***volt, CUDA_VECTOR ***curr, CUDA_VECTOR ***opvi, CUDA_VECTOR ***opvv, unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ) {
	const unsigned int pos[] = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		blockDim.z * blockIdx.z + threadIdx.z
	};

	if(pos[0] >= numLinesX || pos[1] >= numLinesY || pos[2] >= numLinesZ) return;

	UpdateVoltages(volt, curr, opvi, opvv, pos[0], pos[1], pos[2]);
}

__global__ void CurrentKernel(CUDA_VECTOR ***volt, CUDA_VECTOR ***curr, CUDA_VECTOR ***opvi, CUDA_VECTOR ***opvv, unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ) {
	const unsigned int pos[] = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		blockDim.z * blockIdx.z + threadIdx.z
	};

	if(pos[0] >= numLinesX-1 || pos[1] >= numLinesY-1 || pos[2] >= numLinesZ-1) return;

	UpdateCurrents(volt, curr, opvi, opvv, pos[0], pos[1], pos[2]);
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	/*
	Engine_CUDA *instance = this;
	void *args[] = {
		&instance,
		&numTS,
		&iterTS,
		&numLines
	};
	cudaError_t err = cudaLaunchCooperativeKernel((void*)&ManyTS, m_gridDim, m_blockDim, args);
	if(err) throw std::runtime_error("CUDA kernel launch failure: " + std::string(cudaGetErrorString(err)));
	numTS += iterTS;
	*/

	unsigned int endTS = numTS + iterTS;
	for(; numTS < endTS; numTS++)
	{
		VoltageKernel<<<m_gridDim, m_blockDim>>>(volt, curr, Op->vi, Op->vv, numLines[0], numLines[1], numLines[2]);
		cudaError_t err = cudaGetLastError();
		if(err) throw std::runtime_error("CUDA kernel launch failure: " + std::string(cudaGetErrorString(err)));
		CurrentKernel<<<m_gridDim, m_blockDim>>>(volt, curr, Op->iv, Op->ii, numLines[0], numLines[1], numLines[2]);
		err = cudaGetLastError();
		if(err) throw std::runtime_error("CUDA kernel launch failure: " + std::string(cudaGetErrorString(err)));
	}
	cudaDeviceSynchronize();
	return true;
}

