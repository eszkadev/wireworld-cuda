#include <stdio.h>
#include <cuda_runtime.h>
#include <model.hpp>
#include <devicesinfo.h>
#include <omp.h>

int BLOCK_SIZE = 32;
int CELLS_PER_THREAD = 8;
int GRID_SIZE = 16;

#define PART_SIZE BLOCK_SIZE * CELLS_PER_THREAD * GRID_SIZE

int USE_GPU[MAX_GPUS];
int CURRENT_GPU[MAX_GPUS];
Map d_pMap[MAX_GPUS];
Map d_pNewMap[MAX_GPUS];
cudaStream_t stream[MAX_GPUS];

extern "C" DevicesInfo* CUDA_getDevicesList()
{
    DevicesInfo* info = (DevicesInfo*)malloc(sizeof(DevicesInfo));

    cudaGetDeviceCount(&info->nCount);
    info->sNames = (char**)malloc(sizeof(char*) * info->nCount);

    for(int nDevice = 0; nDevice < info->nCount; nDevice++)
    {
        info->sNames[nDevice] = (char*)malloc(sizeof(char) * 256);
        cudaDeviceProp aProperties;
        cudaGetDeviceProperties(&aProperties, nDevice);
        strncpy(info->sNames[nDevice], aProperties.name, 256);
    }

    return info;
}

__global__ void step(Map pOld, Map pNew, unsigned int nWidth, unsigned int nHeight, int nCells)
{
    for(int ix = 0; ix < nCells; ix++)
    for(int iy = 0; iy < nCells; iy++)
    {
        int x = 1 + blockDim.x * blockIdx.x * nCells + threadIdx.x * nCells + ix;
        int y = 1 + blockDim.y * blockIdx.y * nCells + threadIdx.y * nCells + iy;
        if(x < nWidth - 1 && y < nHeight - 1)
        {
            switch(pOld[x * nHeight + y])
            {
            case Head:
                pNew[x * nHeight + y] = Tail;
                break;
            case Tail:
                pNew[x * nHeight + y] = Conductor;
                break;
            case Conductor:
                {
                    unsigned int nHeads = 0;

                    bool bRight = x + 1 < nWidth + 1;
                    bool bLeft = x - 1 >= 0;
                    bool bTop = y + 1 < nHeight;
                    bool bBottom = y - 1 >= 0;

                    if(bRight)
                    {
                        if(pOld[(x + 1) * nHeight + y] == Head)
                            nHeads++;
                        if(bTop && pOld[(x + 1) * nHeight + y + 1] == Head)
                            nHeads++;
                        if(bBottom && pOld[(x + 1) * nHeight + y - 1] == Head)
                            nHeads++;
                    }

                    if(bLeft)
                    {
                        if(pOld[(x - 1) * nHeight + y] == Head)
                            nHeads++;
                        if(bTop && pOld[(x - 1) * nHeight + y + 1] == Head)
                            nHeads++;
                        if(bBottom && pOld[(x - 1) * nHeight + y - 1] == Head)
                            nHeads++;
                    }

                    if(bTop && pOld[x * nHeight + y + 1] == Head)
                        nHeads++;

                    if(bBottom && pOld[x * nHeight + y - 1] == Head)
                        nHeads++;

                    if(nHeads == 1 || nHeads == 2)
                        pNew[x * nHeight + y] = Head;
                    else
                        pNew[x * nHeight + y] = Conductor;
                }
                break;
            default:
                pNew[x * nHeight + y] = Empty;
                break;
            }
        }
    }
}

extern "C" void CUDA_setup()
{
    int nThreadId = omp_get_thread_num();

    cudaFree(d_pMap[nThreadId]);
    cudaFree(d_pNewMap[nThreadId]);

    cudaError_t aError = cudaSuccess;

    aError = cudaMalloc((void**)&d_pMap[nThreadId], (PART_SIZE + 2) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pNewMap[nThreadId], (PART_SIZE + 2) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate new Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }
}

extern "C" void CUDA_exit()
{
    for(int i = 0; i < MAX_GPUS; i++)
    {
        cudaFree(d_pMap[i]);
        cudaFree(d_pNewMap[i]);
    }
}

void CUDA_step_big_data(Model *pModel, int n, int nThreadOrder)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;
    Map pMap = pModel->GetMap();

    for(int i = 0; i < n; ++i)
    {
        for(int nY = 0; nY < nHeight || nY == 0; nY += PART_SIZE)
        {
            int nThreadId = omp_get_thread_num();
            int nThreads = omp_get_num_threads();
            int nStart = 0 + nThreadOrder * PART_SIZE;

            for(int nX = nStart; nX < nWidth || nX == 0; nX += nThreads * PART_SIZE)
            {
                int nLength = (nHeight - nY > PART_SIZE) ? PART_SIZE + 2 : nHeight - nY + 2;
                int nRows = (nWidth - nX > PART_SIZE) ? PART_SIZE + 2 : nWidth - nX + 2;

                if(nRows <= 0 || nLength <= 0)
                    break;

                int nRowDevice = 0;
                for(int nRow = nX; nRowDevice < nRows; nRow += nHeight)
                {
                    aError = cudaMemcpyAsync(d_pMap[nThreadId] + (nRowDevice++) * nLength, pMap + nRow + nY, nLength * sizeof(Cell), cudaMemcpyHostToDevice, stream[nThreadId]);
                    if(aError != cudaSuccess)
                    {
                        fprintf(stderr, "GPU %d: Line: %d, Failed to copy Map[%d,%d,%d] from device to host (error code %s)!\n", nThreadId, __LINE__, nX, nY, nRow, cudaGetErrorString(aError));
                        return;
                    }
                }

                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
                dim3 gridDim(GRID_SIZE, GRID_SIZE, 1);
                printf("GPU %d: CUDA big kernel launch with %dx%d blocks of %dx%d threads\n", nThreadId, gridDim.x, gridDim.y, blockDim.x, blockDim.y);

                step<<<gridDim, blockDim, 0, stream[nThreadId]>>>(d_pMap[nThreadId], d_pNewMap[nThreadId], nRows, nLength, CELLS_PER_THREAD);

                aError = cudaGetLastError();

                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "GPU %d: Failed to launch step kernel (error code %s)!\n", nThreadId, cudaGetErrorString(aError));
                    return;
                }

                nRowDevice = 1;
                for(int nRow = nX; nRowDevice < nRows - 1; nRow += nHeight)
                {
                    aError = cudaMemcpyAsync(pMap + 1 + nHeight + nRow + nY, d_pNewMap[nThreadId] + 1 + (nRowDevice++) * nLength, (nLength - 2) * sizeof(Cell), cudaMemcpyDeviceToHost, stream[nThreadId]);
                    if(aError != cudaSuccess)
                    {
                        fprintf(stderr, "GPU %d: Line: %d, Failed to copy Map[%d,%d,%d] from device to host (error code %s)!\n", nThreadId, __LINE__, nX, nY, nRow, cudaGetErrorString(aError));
                        return;
                    }
                }
            }
        }
        //#pragma omp barrier
    }
}

void CUDA_step_small_data(Model* pModel, int n, int nGPU)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;
    Map pMap = pModel->GetMap();

    aError = cudaMemcpyAsync(d_pMap[nGPU], pMap, nHeight * nWidth * sizeof(Cell), cudaMemcpyHostToDevice, stream[nGPU]);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Line: %d, Failed to copy Map from device to host (error code %s)!\n", __LINE__, cudaGetErrorString(aError));
        return;
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(GRID_SIZE, GRID_SIZE, 1);
    printf("CUDA small kernel launch with %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    for(int i = 0; i < n; ++i)
    {
        step<<<gridDim, blockDim, 0, stream[nGPU]>>>(d_pMap[nGPU], d_pNewMap[nGPU], nWidth, nHeight, CELLS_PER_THREAD);

        aError = cudaGetLastError();

        if(aError != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
            return;
        }

        if(i < n - 1)
        {
            Map pTmp = d_pMap[nGPU];
            d_pMap[nGPU] = d_pNewMap[nGPU];
            d_pNewMap[nGPU] = pTmp;
        }
    }

    aError = cudaMemcpyAsync(pMap, d_pNewMap[nGPU], nHeight * nWidth * sizeof(Cell), cudaMemcpyDeviceToHost, stream[nGPU]);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Line: %d, Failed to copy Map from device to host (error code %s)!\n", __LINE__, cudaGetErrorString(aError));
        return;
    }
}

extern "C" int CUDA_step(Model* pModel, int n)
{
    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;
    int nThreadOrder = 0;

    if(nWidth > PART_SIZE || nHeight > PART_SIZE)
    {
        #pragma omp parallel num_threads(MAX_GPUS)
        {
            int nThreadId = omp_get_thread_num();
            if(USE_GPU[nThreadId])
            {
                cudaSetDevice(nThreadId);

                cudaError_t aError = cudaStreamCreate(&stream[nThreadId]);
                if(aError != cudaSuccess)
                    fprintf(stderr, "Failed to create cuda stream: %d!\n", cudaGetErrorString(aError));

                if(CURRENT_GPU[nThreadId] != USE_GPU[nThreadId])
                {
                    CUDA_setup();
                    CURRENT_GPU[nThreadId] = USE_GPU[nThreadId];
                }

                CUDA_step_big_data(pModel, n, nThreadOrder++);

                cudaStreamDestroy(stream[nThreadId]);
            }
        }
    }
    else
    {
        int nGPU = -1;
        for(int i = 0; i < MAX_GPUS; ++i)
        {
            if(USE_GPU[i] == 1)
            {
                nGPU = i;
                break;
            }
        }

        if(nGPU >= 0)
        {
            printf("Use GPU %d\n", nGPU);

            if(CURRENT_GPU[nGPU] != USE_GPU[nGPU])
            {
                cudaSetDevice(nGPU);
                CUDA_setup();
                CURRENT_GPU[nGPU] = USE_GPU[nGPU];
            }

            cudaError_t aError = cudaStreamCreate(&stream[nGPU]);
            if(aError != cudaSuccess)
                fprintf(stderr, "Failed to create cuda stream: %d!\n", cudaGetErrorString(aError));

            CUDA_step_small_data(pModel, n, nGPU);

            cudaStreamDestroy(stream[nGPU]);
        }
        else
        {
            printf("No GPU selected\n");
        }
    }

    return 0;
}

extern "C" void CUDA_set(int nCells, int nBlock, int nGrid, int* pGPU)
{
    CELLS_PER_THREAD = nCells;
    BLOCK_SIZE = nBlock;
    GRID_SIZE = nGrid;

    for(int i = 0; i < MAX_GPUS; ++i)
        USE_GPU[i] = 0;

    for(int i = 0; i < MAX_GPUS && pGPU[i] >= 0; ++i)
    {
        USE_GPU[pGPU[i]] = 1;
    }

    printf("CUDA settings changed\n");
}
