#include <stdio.h>
#include <cuda_runtime.h>
#include <model.hpp>
#include <devicesinfo.h>

int BLOCK_SIZE = 32;
int CELLS_PER_THREAD = 8;
int GRID_SIZE = 16;

#define PART_SIZE BLOCK_SIZE * CELLS_PER_THREAD * GRID_SIZE

Map d_pMap = NULL;
Map d_pNewMap = NULL;
cudaStream_t stream;

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
    cudaFree(d_pMap);
    cudaFree(d_pNewMap);
    if(stream)
        cudaStreamDestroy(stream);

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
    cudaFree(d_pMap);
    d_pMap = NULL;
    cudaFree(d_pNewMap);
    d_pNewMap = NULL;
    cudaStreamDestroy(stream);
    stream = 0;
    cudaDeviceReset();
}

void CUDA_step_big_data(Model *pModel, int n)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;
    Map pMap = pModel->GetMap();

    for(int i = 0; i < n; ++i)
    {
        for(int nY = 0; nY < nHeight || nY == 0; nY += PART_SIZE)
        {
            for(int nX = 0; nX < nWidth || nX == 0; nX += PART_SIZE)
            {
                int nLength = (nHeight - nY > PART_SIZE) ? PART_SIZE + 2 : nHeight - nY + 2;
                int nRows = (nWidth - nX > PART_SIZE) ? PART_SIZE + 2 : nWidth - nX + 2;

                if(nRows <= 0 || nLength <= 0)
                    break;

                int nRowDevice = 0;
                for(int nRow = nX; nRowDevice < nRows; nRow += nHeight)
                {
                    aError = cudaMemcpyAsync(d_pMap + (nRowDevice++) * nLength, pMap + nRow + nY, nLength * sizeof(Cell), cudaMemcpyHostToDevice, stream);
                    if(aError != cudaSuccess)
                    {
                        fprintf(stderr, "Line: %d, Failed to copy Map[%d,%d,%d] from device to host (error code %s)!\n", __LINE__, nX, nY, nRow, cudaGetErrorString(aError));
                        exit(EXIT_FAILURE);
                    }
                }

                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
                dim3 gridDim(GRID_SIZE, GRID_SIZE, 1);
                printf("CUDA big kernel launch with %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
                fflush(stdout);

                step<<<gridDim, blockDim, 0, stream>>>(d_pMap, d_pNewMap, nRows, nLength, CELLS_PER_THREAD);

                aError = cudaGetLastError();

                if(aError != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
                    exit(EXIT_FAILURE);
                }

                nRowDevice = 1;
                for(int nRow = nX; nRowDevice < nRows - 1; nRow += nHeight)
                {
                    aError = cudaMemcpyAsync(pMap + 1 + nHeight + nRow + nY, d_pNewMap + 1 + (nRowDevice++) * nLength, (nLength - 2) * sizeof(Cell), cudaMemcpyDeviceToHost, stream);
                    if(aError != cudaSuccess)
                    {
                        fprintf(stderr, "Line: %d, Failed to copy Map[%d,%d,%d] from device to host (error code %s)!\n", __LINE__, nX, nY, nRow, cudaGetErrorString(aError));
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
}

void CUDA_step_small_data(Model* pModel, int n)
{
    cudaError_t aError = cudaSuccess;

    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;
    Map pMap = pModel->GetMap();

    aError = cudaMemcpyAsync(d_pMap, pMap, nHeight * nWidth * sizeof(Cell), cudaMemcpyHostToDevice, stream);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Line: %d, Failed to copy Map from device to host (error code %s)!\n", __LINE__, cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(GRID_SIZE, GRID_SIZE, 1);
    printf("CUDA small kernel launch with %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    fflush(stdout);

    for(int i = 0; i < n; ++i)
    {
        step<<<gridDim, blockDim, 0, stream>>>(d_pMap, d_pNewMap, nWidth, nHeight, CELLS_PER_THREAD);

        aError = cudaGetLastError();

        if(aError != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch step kernel (error code %s)!\n", cudaGetErrorString(aError));
            exit(EXIT_FAILURE);
        }

        if(i < n - 1)
        {
            Map pTmp = d_pMap;
            d_pMap = d_pNewMap;
            d_pNewMap = pTmp;
        }
    }

    aError = cudaMemcpyAsync(pMap, d_pNewMap, nHeight * nWidth * sizeof(Cell), cudaMemcpyDeviceToHost, stream);
    if(aError != cudaSuccess)
    {
        fprintf(stderr, "Line: %d, Failed to copy Map from device to host (error code %s)!\n", __LINE__, cudaGetErrorString(aError));
        exit(EXIT_FAILURE);
    }
}

extern "C" int CUDA_step(Model* pModel, int n)
{
    int nWidth = pModel->GetWidth() + 2;
    int nHeight = pModel->GetHeight() + 2;

    if(nWidth > PART_SIZE || nHeight > PART_SIZE)
        CUDA_step_big_data(pModel, n);
    else
        CUDA_step_small_data(pModel, n);

    return 0;
}

extern "C" void CUDA_set(int nCells, int nBlock, int nGrid)
{
    CELLS_PER_THREAD = nCells;
    BLOCK_SIZE = nBlock;
    GRID_SIZE = nGrid;

    CUDA_setup();

    printf("CUDA settings changed\n");
    fflush(stdout);
}
