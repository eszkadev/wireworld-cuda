#include <stdio.h>
#include <cuda_runtime.h>
#include <model.hpp>

#define PART_SIZE 128
#define BLOCK_X 32
#define BLOCK_Y 32

Cell* d_pMap = NULL;
Cell* d_pNextMap = NULL;
Cell* d_pNewMap = NULL;

__global__ void step(Cell* pOld, Cell* pNew, unsigned int nWidth, unsigned int nHeight)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // first row is the last row of block neighbour
    x++;
    // each row have additional Cell at the end
    nHeight++;

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

extern "C" void CUDA_setup()
{
    cudaError_t aError = cudaSuccess;

    aError = cudaMalloc((void**)&d_pMap, (PART_SIZE + 1) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pNewMap, (PART_SIZE + 1) * (PART_SIZE + 2) * sizeof(Cell));

    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate new Map (error code %s)!\n", cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    aError = cudaMalloc((void**)&d_pNextMap, (PART_SIZE + 1) * (PART_SIZE + 2) * sizeof(Cell));

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

    static Cell aEmpty[PART_SIZE + 1] = { Empty };

    int nCounter = 0;

    for(int nY = 0; nY < nHeight; nY += PART_SIZE)
    {
        for(int nX = 0; nX < nWidth; nX += PART_SIZE - 1)
        {
            if(nX == 0)
            {
                // Fill the first row with Empty cells
                aError = cudaMemcpy(d_pMap, aEmpty, (PART_SIZE + 1) * sizeof(Cell), cudaMemcpyHostToDevice);
                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Line: %d, Part: %d Failed to copy Map[%d] from host to device (error code %s)!\n", __LINE__, 0, 0, cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }

                for(nCounter = 0; nCounter < PART_SIZE + 1 && nCounter < nWidth; ++nCounter)
                {
                    aError = cudaMemcpy(d_pMap + (nCounter * (PART_SIZE + 1)), pMap[nCounter] + nY, (PART_SIZE + 1) * sizeof(Cell), cudaMemcpyHostToDevice);

                    if(aError != cudaSuccess)
                    {
                        fprintf(stderr, "Line: %d, Part: %d Failed to copy Map[%d] from host to device (error code %s)!\n", __LINE__, 0, nCounter, cudaGetErrorString(aError));
                        exit(EXIT_FAILURE);
                    }
                }
            }

            dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
            dim3 gridDim((PART_SIZE + BLOCK_X - 1) / BLOCK_X, (PART_SIZE + BLOCK_Y - 1) / BLOCK_Y, 1);
            printf("CUDA kernel launch with %dx%d blocks of %dx%d threads on [%d, %d]\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, nX, nY);

            step<<<gridDim, blockDim>>>(d_pMap, d_pNewMap, PART_SIZE, PART_SIZE);
            aError = cudaGetLastError();

            if(aError != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
                exit(EXIT_FAILURE);
            }

            nCounter = 0;
            for(int i = nX; i < nX + PART_SIZE + 2 && i < nWidth; ++i)
            {
                if(i < nX + PART_SIZE)
                    aError = cudaMemcpy(pMap[i] + nY, d_pNewMap + ((nCounter + 1) * (PART_SIZE + 1)), PART_SIZE * sizeof(Cell), cudaMemcpyDeviceToHost);

                if(i + PART_SIZE - 1 < nWidth)
                    aError = cudaMemcpy(d_pNextMap + (nCounter * (PART_SIZE + 1)), pMap[i + (PART_SIZE - 2)] + nY, (PART_SIZE + 1) * sizeof(Cell), cudaMemcpyHostToDevice);

                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Line: %d, Part: [%d,%d] Failed to copy Map[%d] from host to device (error code %s)!\n", __LINE__, nX, nY, i, cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }

                nCounter++;
            }

            Cell* pTmp = d_pMap;
            d_pMap = d_pNextMap;
            d_pNextMap = pTmp;
        }
    }

    return 0;
}

