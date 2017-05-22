#include <stdio.h>
#include <cuda_runtime.h>
#include <model.hpp>

#define PART_SIZE 2048
#define BLOCK_X 32
#define BLOCK_Y 32
#define GRID_X 16 //(PART_SIZE + BLOCK_X - 1) / BLOCK_X
#define GRID_Y 16 //(PART_SIZE + BLOCK_Y - 1) / BLOCK_Y

Cell* d_pMap = NULL;
Cell* d_pNewMap = NULL;
cudaStream_t stream;

__global__ void step(Cell* pOld, Cell* pNew, unsigned int nWidth, unsigned int nHeight)
{
    // each row have additional Cell at the end
    nHeight += 2;
    nWidth += 2;

    for(int ix = 0; ix < PART_SIZE/GRID_X/BLOCK_X; ix++)
    for(int iy = 0; iy < PART_SIZE/GRID_X/BLOCK_X; iy++)
    {
        int offset = PART_SIZE/GRID_X/BLOCK_X;
        int x = 1 + blockDim.x * blockIdx.x * offset + threadIdx.x * offset + ix;
        int y = 1 + blockDim.y * blockIdx.y * offset + threadIdx.y * offset + iy;
        if(x < nWidth && y < nHeight)
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
    cudaError_t aError = cudaSuccess;

    aError = cudaStreamCreate(&stream);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to create cuda stream: %d!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pMap, (PART_SIZE + 2) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pNewMap, (PART_SIZE + 2) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate new Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }
}

extern "C" void CUDA_exit()
{
    cudaError_t aError = cudaSuccess;

    aError = cudaFree(d_pMap);

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaFree(d_pNewMap);

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device new Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaStreamDestroy(stream);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to destroy cuda stream (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    aError = cudaDeviceReset();

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }
}

extern "C" int CUDA_step(Model* pModel, int n)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth();
    int nHeight = pModel->GetHeight();
    Map pMap = pModel->GetMap();

    // TODO sliding window
    //for(int nY = 0; nY < nHeight; nY += PART_SIZE)
    {
    //    for(int nX = 0; nX < nWidth; nX += PART_SIZE)
        {
            for(int i = 0; i < nWidth + 2; i++)
            {
                aError = cudaMemcpyAsync(d_pMap + i*(nHeight+2), pMap[i], (nHeight + 2) * sizeof(Cell), cudaMemcpyHostToDevice, stream);
                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Line: %d, Failed to copy Map[%d] from device to host (error code %s)!\n", __LINE__, i, cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }
            }


            dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
            dim3 gridDim(GRID_X, GRID_Y, 1);
            printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

            for(int i = 0; i < n; ++i)
            {
                step<<<gridDim, blockDim, 0, stream>>>(d_pMap, d_pNewMap, nWidth, nHeight);

                aError = cudaGetLastError();

                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }

                if(i + 1 < n)
                {
                    Cell* pTmp = d_pMap;
                    d_pMap = d_pNewMap;
                    d_pNewMap = pTmp;
                }
            }

            for(int i = 0; i < nWidth + 2; i++)
            {
                aError = cudaMemcpyAsync(pMap[i], d_pNewMap + (nHeight+2)*i, (nHeight + 2) * sizeof(Cell), cudaMemcpyDeviceToHost, stream);
                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Line: %d, Failed to copy Map[%d] from device to host (error code %s)!\n", __LINE__, i, cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    return 0;
}

