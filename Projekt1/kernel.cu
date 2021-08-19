#include "kernel.cuh"
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/count.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <algorithm>


__device__ int CalculateCellId(float x, float y, Parameters_t& params)
{
	int id = ((y * params.width * params.height) / (float)params.particle_size + x * params.width) / (float)params.particle_size;
	return id;
}

__device__ void CalculateNewVelocity(int i, int j, NewVelocities_t& newvel, Particles_t& particles, Parameters_t& params)
{
	//jesli i jest rowne j to nie rozwazamy zderzenia
	if (i == j)
	{
		return;
	}
	//zastosowanie wzoru na zderzenie
	float masspart = 2 * particles.mass[j] / (particles.mass[i] + particles.mass[j]);
	float2 v2 = particles.GetVelocity(j);
	//float2 v1 = particles.GetVelocity(i);
	float2 v1 = make_float2(newvel.nvx[i], newvel.nvy[i]);
	float2 x2 = particles.GetPosition(j);
	float2 x1 = particles.GetPosition(i);
	float2 dx = x1 - x2;
	float2 dv = v1 - v2;
	float ldx = length(dx);
	//jesli czasteczki sie nie stykaja nie modyfikujemy predkosci
	if (fabsf(dx.x) < params.particle_size && fabsf(dx.y) < params.particle_size)
	{
		float2 newV = masspart * dot(dx, dv) / (ldx * ldx) * dx;
		newvel.nvx[i] = newvel.nvx[i] - newV.x;
		newvel.nvy[i] = newvel.nvy[i] - newV.y;
	}
	return;
}

__global__ void gravity_and_damp_kernel(Particles_t particles, float dt, int* cellsx, int* cellsy, Parameters_t params)
{
	int i = GetID();
	//zabezpieczenie na wypadake gdy jest wiecej watkow niz elementow w tablicach
	if (i >= params.particles_count)
		return;
	
	//obliczenie wplywu przyspieszenia grawitacyjnego
	particles.vy[i] += params.g_const * dt;
	//przesuniecie czasteczek
	particles.x[i] += particles.vx[i] * dt;
	particles.y[i] += particles.vy[i] * dt;
	//rozpatrzenie sytuacji gdy czasteczka jest powyzej gornej sciany lub ponizej dolnej
	if (particles.y[i] <= 0 || particles.y[i] >= 1.0)
	{
		//odwrocenie kierunku wektora predkosci i jego zmniejszenie ( lub zwiekszenie) o wspolczynnik 
		particles.vy[i] *= -params.dumping_factor;

		if (particles.y[i] < 0)
		{
			//jesli czasteczki maja odbijac sie od dolnej scianki ich pozycja w osi y=0 jesli nie to ich y=1
			particles.y[i] = params.inf_loop;
		}
		else if (particles.y[i] > 1)
		{
			//faza sp³aszczenia - symulacja czasu spedzanego na odksztalceniu po kolizji
			particles.y[i] = 1;
		}
	}
	//analogicznie dla wpsolrzednej x
	if (particles.x[i] <= 0 || particles.x[i] >= 1.0)
	{
		particles.vx[i] *= -params.dumping_factor;

		if (particles.x[i] < 0)
		{
			//faza sp³aszczenia
			particles.x[i] = 0;
		}
		else if (particles.x[i] > 1)
		{
			//faza sp³aszczenia
			particles.x[i] = 1;
		}
	}

	//wypelnienie tablic pomocniczej 
	//cells_x - komorka w ktorej jest czasteczka "i"
	//cells_y - numer czasteczki = i
	cellsx[i] = CalculateCellId(particles.x[i], particles.y[i], params);
	cellsy[i] = i;
}

__global__ void fill_startend(int2* startend, Parameters_t params)
{
	int i = GetID();
	//zabezpieczenie przed nielegalnym dostepem do pamieci
	if (i < params.height * params.width)
	{
		startend[i].x = -1;
		startend[i].y = -1;
	}
}

__global__ void find_startend_kernel(int* cellsx, int* cellsy, int2* startend, Parameters_t params)
{
	int i = GetID();
	//zabezpieczenie przciwko niepoprawnemu dostepowi
	if (i > 0 && i < params.particles_count)
	{
		//jesli w poprzedniej komorce jest inna wartosc niz w obecnej to oznacza to ze lista "cell'a" o id z poprzedniej komorki sie skonczyla
		//	a zaczela sie lista nowego "cell'a, jako ze dane w cellsx sa posortowane
		if (cellsx[i - 1] != cellsx[i])
		{
			startend[cellsx[i]].x = i;
			startend[cellsx[i - 1]].y = i;
		}
	}
	//wypelnienie danych dla krancowych komorek
	if (i == 0)
	{
		startend[cellsx[0]].x = 0;
		startend[cellsx[params.particles_count - 1]].y = params.particles_count;
	}
}

__global__ void collision_kernel(Particles_t particles, int* cellsx, int* cellsy, int2* startend, NewVelocities_t newVelocities, Parameters_t params)
{
	int i = GetID();
	
	//zabezpieczenie przeciwko nieprawidlowemu dostepowi do pamieci
	if (i < params.particles_count)
	{
		//obliczenie komorki w ktorej jest czasteczka
		int cellid = CalculateCellId(particles.x[i], particles.y[i], params);
		//wypelnienie tablicy nowych predkosci oebcna predkoscia
		newVelocities.nvx[i] = particles.vx[i];
		newVelocities.nvy[i] = particles.vy[i];
		
		//komorki do sprawdzenia
		//-------------
		//| 0 | 1 | 2 | 
		//-------------
		//| 3 | 4 | 5 | 
		//-------------
		//| 6 | 7 | 8 | 
		//-------------
		//Czasteczka znajduje sie w srodkowej komorce (nr 4), musi zatem sprawdzic ta komorke i 8 lezacych na okolo niej

		//cell 4 
		//position (x,y)
		//sprawdzana komorka o:
		//id = cellid
		if (cellid>=0&&startend[cellid].x != -1 && startend[cellid].y != -1)
		{
			for (int j = startend[cellid].x; j < startend[cellid].y; j++)
			{
				if (cellsy[j] != i && cellsx[j] == cellid)
				{
					CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
				}
			}
		}

		//cell 0
		//position (x-1,y+1)
		//sprawdzana komorka o:
		//id = cellid - 1 + params.width / params.particle_size
		if (cellid - 1 + params.width / params.particle_size < params.MaxCellId()&& cellid - 1 + params.width / params.particle_size>=0)
		{
			if (startend[cellid - 1 + params.width / params.particle_size].x != -1 && startend[cellid - 1 + params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid - 1 + params.width / params.particle_size].x; j < startend[cellid - 1 + params.width / params.particle_size].y; j++)
				{
					if (cellsy[j] != i)
					{
						CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
					}
				}
			}
		}

		////cell 1
		////position (x,y+1)
		//sprawdzana komorka o:
		//id = cellid + params.width / params.particle_size
		if (cellid + params.width / params.particle_size < params.MaxCellId()&& cellid + params.width / params.particle_size>=0)
		{
			if (startend[cellid + params.width / params.particle_size].x != -1 && startend[cellid + params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid + params.width / params.particle_size].x; j < startend[cellid + params.width / params.particle_size].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 2
		////position (x+1,y+1)
		//sprawdzana komorka o:
		//id = cellid + 1 + params.width / params.particle_size
		if (cellid + 1 + params.width / params.particle_size < params.MaxCellId()&& cellid + 1 + params.width / params.particle_size>=0)
		{
			if (startend[cellid + 1 + params.width / params.particle_size].x != -1 && startend[cellid + 1 + params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid + 1 + params.width / params.particle_size].x; j < startend[cellid + 1 + params.width / params.particle_size].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 3
		////position (x-1,y)
		//sprawdzana komorka o:
		//id = cellid - 1
		if (cellid - 1 < params.MaxCellId()&& cellid - 1>=0)
		{
			if (startend[cellid - 1].x != -1 && startend[cellid - 1].y != -1)
			{
				for (int j = startend[cellid - 1].x; j < startend[cellid - 1].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 5
		////position (x+1,y)
		//sprawdzana komorka o:
		//id = cellid + 1
		if (cellid + 1 < params.MaxCellId()&& cellid + 1>=0)
		{
			if (startend[cellid + 1].x != -1 && startend[cellid + 1].y != -1)
			{
				for (int j = startend[cellid + 1].x; j < startend[cellid + 1].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		//cell 6
		//sprawdzana komorka o:
		//id = cellid - 1 - params.width / params.particle_size
		if (cellid - 1 - params.width / params.particle_size < params.MaxCellId()&& cellid - 1 - params.width / params.particle_size>=0)
		{
			if (startend[cellid - 1 - params.width / params.particle_size].x != -1 && startend[cellid - 1 - params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid - 1 - params.width / params.particle_size].x; j < startend[cellid - 1 - params.width / params.particle_size].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		//cell 7
		//position (x,y-1)
		//sprawdzana komorka o:
		//id = cellid - params.width / params.particle_size
		if (cellid - params.width / params.particle_size < params.MaxCellId()&& cellid - params.width / params.particle_size>=0)
		{
			if (startend[cellid - params.width / params.particle_size].x != -1 && startend[cellid - params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid - params.width / params.particle_size].x; j < startend[cellid - params.width / params.particle_size].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		//cell 8
		//position (x+1,y-1)
		//sprawdzana komorka o:
		//id = cellid + 1 - params.width / params.particle_size
		if (cellid + 1 - params.width / params.particle_size < params.MaxCellId()&& cellid + 1 - params.width / params.particle_size>=0)
		{
			if (startend[cellid + 1 - params.width / params.particle_size].x != -1 && startend[cellid + 1 - params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid + 1 - params.width / params.particle_size].x; j < startend[cellid + 1 - params.width / params.particle_size].y; j++)
				{
					if (j < params.particles_count)
					{
						if (cellsy[j] != i)
						{
							CalculateNewVelocity(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
	}
}

__global__ void calculate_new_velocity_kernel(Particles_t particles, NewVelocities_t newvel, Parameters_t params)
{
	int i = GetID();
	//zabezpieczenie przed nieprawidlowym dostepem do pamieci
	if (i < params.particles_count)
	{
		//nadpisanie starej predkosci jesli jest ona wieksza od EPS
		if (fabsf(newvel.nvx[i]) > EPS)
		{
			//zastosowanie wspolczynnika zmniejszajacego predkosc po kolizji
			particles.vx[i] = params.collision_dumping_factor * newvel.nvx[i];
		}
		if (fabsf(newvel.nvy[i]) > EPS)
		{
			particles.vy[i] = newvel.nvy[i];
		}
	}
}

void PrepareCuda(Particles_t& cpu, Particles_t& dev_part, NewVelocities_t& dev_newvel, Parameters_t& params, Cells_t& dev_cells)
{
	cudaError_t cudaStatus;

	//Alokacja pamieci na tablice na GPU dla struktury opisujacej cz¹steczki i przekopiowanie do nich danych z CPU
	float* dev_x, * dev_y, * dev_vx, * dev_vy, * dev_mass;
	MallocCudaFloat(cudaStatus, dev_x, params.particles_count, sizeof(float), cpu.x);
	MallocCudaFloat(cudaStatus, dev_y, params.particles_count, sizeof(float), cpu.y);
	MallocCudaFloat(cudaStatus, dev_vx, params.particles_count, sizeof(float), cpu.vx);
	MallocCudaFloat(cudaStatus, dev_vy, params.particles_count, sizeof(float), cpu.vy);
	MallocCudaFloat(cudaStatus, dev_mass, params.particles_count, sizeof(float), cpu.mass);
	//przypisanie wskaznikow
	dev_part.size = params.particles_count;
	dev_part.x = dev_x;
	dev_part.y = dev_y;
	dev_part.vx = dev_vx;
	dev_part.vy = dev_vy;
	dev_part.mass = dev_mass;

	//Alokacja pamieci na tablice zawierajace nowe predkosci
	float* dev_nvx, * dev_nvy;
	MallocCudaFloat(cudaStatus, dev_nvx, params.particles_count, sizeof(float), nullptr);
	MallocCudaFloat(cudaStatus, dev_nvy, params.particles_count, sizeof(float), nullptr);
	//przypisanie wskaznikow
	dev_newvel.nvx = dev_nvx;
	dev_newvel.nvy = dev_nvy;

	//Alokacja tablic pomocniczych
	int* dev_cells_x, * dev_cells_y;
	int2* dev_startend;
	MallocCudaInt(cudaStatus, dev_cells_x, params.particles_count, sizeof(int), nullptr);
	MallocCudaInt(cudaStatus, dev_cells_y, params.particles_count, sizeof(int), nullptr);
	MallocCudaInt2(cudaStatus, dev_startend, params.width * params.height, sizeof(int2), nullptr);
	//przypisanie wskaznikow
	dev_cells.cells_x = dev_cells_x;
	dev_cells.cells_y = dev_cells_y;
	dev_cells.startend = dev_startend;
}

void FreeCudaArrays(Particles_t& dev_part, NewVelocities_t& dev_newvel, Cells_t& dev_cells)
{
	cudaFree(dev_part.mass);
	cudaFree(dev_part.vx);
	cudaFree(dev_part.vy);
	cudaFree(dev_part.x);
	cudaFree(dev_part.y);
	cudaFree(dev_newvel.nvx);
	cudaFree(dev_newvel.nvy);
	cudaFree(dev_cells.cells_x);
	cudaFree(dev_cells.cells_y);
	cudaFree(dev_cells.startend);
}

float RunCuda(Particles_t& cpu_part, Particles_t& dev_part, NewVelocities_t& dev_newvel, float t, Parameters_t& params, Cells_t& dev_cells, float& cpytime)
{
	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaError_t cudaStatus;
	

	//Stowrzenie timerow i rozpoczecie liczenia czasu wykonania
	cudaStatus=cudaEventCreate(&start);
	CheckForErrors(cudaStatus, "create start event");
	cudaStatus=cudaEventCreate(&stop);
	CheckForErrors(cudaStatus, "create stop event");
	cudaStatus = cudaEventRecord(start, 0);
	CheckForErrors(cudaStatus, "event record start");


	//Obliczenie wymiarow blokow dla kerneli dzialajacych na tablicach rozmiaru params.particles_count
	dim3 block_dim, grid_dim;
	CalculateGridBlockDimensions(block_dim, grid_dim, params.particles_count);


	//Przesuniecie czasteczek wykorzystujac ich predkosc z poprzednich wywolan funckji oraz uwzgledniajac przyspieszenie grawitacyjne,
	//	a takze wypelnienie tablic pomocniczych opisujacych miejsce czasteczek w komorkach (cells)
	gravity_and_damp_kernel << <grid_dim, block_dim >> > (dev_part, t, dev_cells.cells_x, dev_cells.cells_y, params);
	CheckForErrorsAndSynchronize(cudaStatus, "gravity");


	//Sortowanie tablic pomocniczych po numerach komorek w ktorych znajduja sie czasteczki - umozliwia to poxniej stworzenie tablicy(grida)
	//	przyspieszajacej obliczanie zderzen 
	thrust::sort_by_key(
		thrust::device_ptr<int>(dev_cells.cells_x),
		thrust::device_ptr<int>(dev_cells.cells_x + params.particles_count),
		thrust::device_ptr<int>(dev_cells.cells_y));


	//Znalezienie poczatkowej i koncowej komorki tablicy cells_X zawierajacej czasteczki z danej komorki(cella)
	dim3 pixels_grid_dim, pixels_block_dim;
	CalculateGridBlockDimensions(pixels_block_dim, pixels_grid_dim, params.width * params.height);
	//Wype³nienie tablicy danymi (-1,-1) w celu nadpisania danych z poprzedniej iteracji
	fill_startend << <pixels_grid_dim, pixels_block_dim >> > (dev_cells.startend, params);
	CheckForErrorsAndSynchronize(cudaStatus, "fill");
	//Obliczenie poczatku i konca komorek
	find_startend_kernel << <grid_dim, block_dim >> > (dev_cells.cells_x, dev_cells.cells_y, dev_cells.startend, params);
	CheckForErrorsAndSynchronize(cudaStatus, "find");


	//Obliczenie kolizji i nowych predkosci
	collision_kernel << <grid_dim, block_dim >> > (dev_part, dev_cells.cells_x, dev_cells.cells_y, dev_cells.startend, dev_newvel, params);
	CheckForErrorsAndSynchronize(cudaStatus, "collision");
	//Zastapienie starych predkosci nowymi
	calculate_new_velocity_kernel << <grid_dim, block_dim >> > (dev_part, dev_newvel, params);
	CheckForErrorsAndSynchronize(cudaStatus, "velocity");


	//Zatrzymanie liczenia czasu
	cudaStatus= cudaEventRecord(stop, 0);
	CheckForErrors(cudaStatus, "record event stop");
	cudaStatus= cudaEventSynchronize(stop);
	CheckForErrors(cudaStatus, "event synchronize");
	cudaStatus = cudaEventElapsedTime(&elapsed, start, stop);
	CheckForErrors(cudaStatus, "elapsed time");
	cudaStatus = cudaEventDestroy(start);
	CheckForErrors(cudaStatus, "start event destroy");
	cudaStatus = cudaEventDestroy(stop);
	CheckForErrors(cudaStatus, "stop event destroy");


	//Przekopiowanie danych o pozycji czasteczek na CPU i zmierzenie czasu tego kopiowania
	std::clock_t c_start = std::clock();
	MemcpyCudaParticles(cudaStatus, dev_part, cpu_part, params);
	std::clock_t c_end = std::clock();
	cpytime = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

	return elapsed;
}

void MemcpyCudaParticles(cudaError_t& cudaStatus, Particles_t& dev_part, Particles_t& cpu_part, Parameters_t& params)
{
	//kopiowanie pozycji x
	cudaStatus = cudaMemcpy(cpu_part.x, dev_part.x, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//kopiowanie pozycji y
	cudaStatus = cudaMemcpy(cpu_part.y, dev_part.y, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
}

void CalculateGridBlockDimensions(dim3& block_dim, dim3& grid_dim, int size)
{
	block_dim.x = 1024;
	block_dim.y = 1;
	block_dim.z = 1;
	grid_dim.x = size / (block_dim.x * block_dim.y * block_dim.z) + 1;
	grid_dim.y = 1;
	grid_dim.z = 1;
}

int SetCudaDevice(bool& retflag)
{
	retflag = true;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		retflag = true;
		return 1;
	}
	retflag = false;
	return 0;
}

void CheckForErrorsAndSynchronize(cudaError_t& cudaStatus, const char* str)
{
	//sprawdzenie bledu uruchomienia
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s kernel launch failed: %s\n", str, cudaGetErrorString(cudaStatus));
	}
	//synchronizacja i sprawdzenie bledu synchronizacji
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s!\n", cudaStatus, str);
	}
}

void CheckForErrors(cudaError_t& cudaStatus, const char* str)
{
	//sprawdzenie kodu bledu
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda returned error code %d after %s!\n", cudaStatus, str);
	}
}

void MemcpyCuda(cudaError_t& cudaStatus, float* px, float* py, float* pvx, float* pvy, float* x, float* y, float* vx, float* vy, Parameters_t& params)
{
	//kopiowanie pozycji x
	cudaStatus = cudaMemcpy(x, px, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//kopiowanie pozycji y
	cudaStatus = cudaMemcpy(y, py, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//kopiowanie predkosci x
	cudaStatus = cudaMemcpy(vx, pvx, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//kopiowanie predkosci y
	cudaStatus = cudaMemcpy(vy, pvy, params.particles_count * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
}

void MallocCudaFloat(cudaError_t& cudaStatus, float*& tab, int count, int size, float* lasttab = nullptr)
{
	//alokacja
	cudaStatus = cudaMalloc((void**)&tab, count * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	//kopiowanie
	if (lasttab != nullptr)
	{
		cudaStatus = cudaMemcpy(tab, lasttab, count * size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
	}
}

void MallocCudaInt(cudaError_t& cudaStatus, int*& tab, int count, int size, int* lasttab = nullptr)
{
	//alokacja
	cudaStatus = cudaMalloc((void**)&tab, count * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	//kopiowanie
	if (lasttab != nullptr)
	{
		cudaStatus = cudaMemcpy(lasttab, tab, count * size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			//goto Error;
		}
	}
}

void MallocCudaInt2(cudaError_t& cudaStatus, int2*& tab, int count, int size, int2* lasttab = nullptr)
{
	//alokacja
	cudaStatus = cudaMalloc((void**)&tab, count * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	//kopiowanie
	if (lasttab != nullptr)
	{
		cudaStatus = cudaMemcpy(lasttab, tab, count * size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
	}
}

void FreeCudaArray(void* pointer)
{
	cudaFree(pointer);
}