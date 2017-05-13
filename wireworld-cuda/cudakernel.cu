#include <stdio.h>
#include <cuda_runtime.h>
#include <model.hpp>

#define PART_SIZE 128
#define BLOCK_X 32
#define BLOCK_Y 32

Cell* d_pMap = NULL;
Cell* d_pNewMap = NULL;

__global__ void step(Cell* pOld, Cell* pNew, unsigned int nWidth, unsigned int nHeight)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

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

            bool bRight = x + 1 < nWidth;
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

extern "C" void CUDA_setup(Model* pModel)
{
    cudaError_t aError = cudaSuccess;
    int nHeight = pModel->GetHeight();

    aError = cudaMalloc((void**)&d_pMap, nHeight * PART_SIZE * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pNewMap, nHeight * PART_SIZE * sizeof(Cell));

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

    // Reset the device and exit
    aError = cudaDeviceReset();

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }
}

extern "C" int CUDA_step(Model* pModel)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth();
    int nHeight = pModel->GetHeight();
    Map pMap = pModel->GetMap();

    int nCounter = 0;
    for(int i = 0; i < PART_SIZE && i < nWidth; ++i)
    {
        aError = cudaMemcpy(d_pMap + (nCounter * nHeight), pMap[i], nHeight * sizeof(Cell), cudaMemcpyHostToDevice);

        if(aError != cudaSuccess)
        {
            fprintf(stderr, "Part: %d Failed to copy Map[%d] from host to device (error code %s)!\n", 0, i, cudaGetErrorString(aError));
            exit(EXIT_FAILURE);
        }

        nCounter++;
    }

    for(int nPart = 0; nPart < nWidth; nPart += PART_SIZE)
    {
        dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
        dim3 gridDim((PART_SIZE + BLOCK_X - 1) / BLOCK_X, (nHeight + BLOCK_Y - 1) / BLOCK_Y, 1);
        printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        step<<<gridDim, blockDim>>>(d_pMap, d_pNewMap, PART_SIZE, nHeight);
        aError = cudaGetLastError();

        if(aError != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
            exit(EXIT_FAILURE);
        }

        nCounter = 0;
        for(int i = nPart; i < nPart + PART_SIZE && i < nWidth; ++i)
        {
            aError = cudaMemcpy(pMap[i], d_pNewMap + (nCounter * nHeight), nHeight * sizeof(Cell), cudaMemcpyDeviceToHost);
            if(i + PART_SIZE < nWidth)
                aError = cudaMemcpy(d_pMap + (nCounter * nHeight), pMap[i + PART_SIZE], nHeight * sizeof(Cell), cudaMemcpyHostToDevice);

            if(aError != cudaSuccess)
            {
                fprintf(stderr, "Part: %d Failed to copy Map[%d] from device to host (error code %s)!\n", nPart, i, cudaGetErrorString(aError));
                exit(EXIT_FAILURE);
            }

            nCounter++;
        }
    }

    return 0;
}

