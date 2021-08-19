#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.h"


#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include <GL/freeglut.h>
#include <cudaGL.h>

#include "constants.h"
#include "macros.h"

#include <helper_math.h>

/// <summary>
/// Struktura opisujace czasteczki
/// </summary>
struct Particles_t
{
	//liczba czasteczek
	int size;
	//tablica mas czasteczek
	float* mass;
	//tablica wartosci "x" polozenia czasteczek
	float* x;
	//tablica wartosci "y" polozenia czasteczek
	float* y;
	//tablica wartosci "x" predkosci czasteczek
	float* vx;
	//tablica wartosci "y" predkosci czasteczek
	float* vy;

	//funckja zwracajaca predkosc czasteczki "i" jako zmienna float2, funkcji mozna uzywac zarowno na GPU jak i CPU
	__host__ __device__ float2 GetVelocity(int i)
	{
		return make_float2(vx[i], vy[i]);
	}

	//funckja zwracajaca polozenie czasteczki "i" jako zmienna float2, funkcji mozna uzywac zarowno na GPU jak i CPU
	__host__ __device__ float2 GetPosition(int i)
	{
		return make_float2(x[i], y[i]);
	}
};

//struktura zawierajaca nowe wartosci predkosci kazdej z czasteczek
struct NewVelocities_t
{
	float* nvx;
	float* nvy;
};

//struktura zawierajaca parametry symulacji
struct Parameters_t
{
	//zmienna oznaczajacy czy czasteczki odbijaja sie od dna okna,
	// czy w przypadku jego osiagniecia pojawiaja sie na wysokosci gornej krawedzi okna
	bool inf_loop;
	//wysokosc okna animacji
	int height;
	//szerokosc okna animacji
	int width;
	//liczba czasteczek w symulacji
	int particles_count;
	//rozmiar czasteczki (srednica lub bok kwadratu)
	int particle_size;
	float g_const;
	//czynnik przez ktory jest przemnazana wartosc predkosci x i y po zderzeniu ze sciankami okna 
	float dumping_factor;
	//czynnik przez ktory jest przemnazana wartosc predkosci x po zderzeniu (ma symulowac zamiane energi kinetycznej 
	// na energie cieplna/spowodowanie oksztalcen przy zderzeniu przez uklad czasteczek sie zderzajacych
	float collision_dumping_factor;
	//wartosc epsilona na potrzeby obliczen
	float eps;

	//Maksymalne id komorki w tablicy zderzen
	__device__ int MaxCellId()
	{
		return (int)(width * height / (float)particle_size / (float)particle_size);
	}
};

//struktura pomocnicza wykorzystywana do budowy tablicy zderzen
struct Cells_t
{
	//numer komorki
	int* cells_x;
	//numer czasteczki
	int* cells_y;
	//przedzial [x,y) z tablicy cells_x dla ktorej w cells_y jest dane id danej komorki w tablicy cellsx
	//iterator  cells x
	//0			5
	//1			6
	//2			6
	//3			6
	//4			10
	//koniec
	//iterator  startend.x  startend.y
	//0			-1			-1
	//1			-1			-1
	//...
	//5			0			1
	//6			1			4
	//7			-1			-1
	//...
	//10		4			5
	int2* startend;
};

//Funckje wolane z CPU

/// <summary>
/// Funkcja alokujaca pamiec GPU i kopiujaca dane z CPU na GPU
/// </summary>
/// <param name="cudaStatus">zmienna zawierajaca kod bledu</param>
/// <param name="dev_part">struktura zawierajaca tablice danych(czasteczek) alokowane na GPU</param>
/// <param name="cpu_part">struktura zawierajaca tablice danych(czasteczek) alokowane na CPU</param>
/// <param name="params">struktura zawierajaca parametry symulacji</param>
void MemcpyCudaParticles(cudaError_t& cudaStatus, Particles_t& dev_part, Particles_t& cpu_part, Parameters_t& params);
/// <summary>
/// Funkcja wybierajaca GPU 
/// </summary>
/// <param name="retflag">flaga bledu</param>
/// <returns></returns>
int SetCudaDevice(bool& retflag);
/// <summary>
/// Funkcja zwalniajaca pamiec GPU
/// </summary>
/// <param name="dev_part">struktura zawierajaca tablice danych(czasteczek) alokowane na GPU</param>
/// <param name="dev_newvel">struktura zawierajaca tablice nowych predkosci alokowana na GPU</param>
/// <param name="dev_cells">struktura pomocnicza wykorzystywana przy budowie tablicy zderzen zawierajaca tablice alokowane na GPU</param>
void FreeCudaArrays(Particles_t& dev_part, NewVelocities_t& dev_newvel, Cells_t& dev_cells);
/// <summary>
/// Funkcja przygotowujaca dane do symulacji
/// </summary>
/// <param name="cpu">czasteczki alokowane na CPU</param>
/// <param name="dev_part">czasteczki alokowane na GPU</param>
/// <param name="dev_newvel">nowe predkosci alokowane na GPU</param>
/// <param name="params">parametry symulacji</param>
/// <param name="dev_cells">pomocnicze tablice na GPU do budowy tablciy zderzen</param>
void PrepareCuda(Particles_t& cpu, Particles_t& dev_part, NewVelocities_t& dev_newvel, Parameters_t& params, Cells_t& dev_cells);
/// <summary>
/// Funkcja obliczajaca nowa pozycje czasteczek
/// </summary>
/// <param name="cpu_part">czasteczki alokowane na CPU</param>
/// <param name="dev_part">czasteczki alokowane na GPU</param>
/// <param name="dev_newvel">nowe predkosci czasteczek alokowane na GPU</param>
/// <param name="t">czas trawnia ruchu</param>
/// <param name="params">parametry symulacji</param>
/// <param name="dev_cells">pomocnicze tablice na GPU do budowy tablciy zderzen</param>
/// <param name="cpytime">czas kopiowania danych z CPU na CPU</param>
/// <returns>czas obliczeñ na GPU</returns>
float RunCuda(Particles_t& cpu_part, Particles_t& dev_part, NewVelocities_t& dev_newvel, float t, Parameters_t& params, Cells_t& dev_cells, float& cpytime);
/// <summary>
/// Funkcja obliczajaca rozmiary siatki i blokow na podstawie wielkosci danych, wymiary bloku sa postaci(1024,1,1)
/// </summary>
/// <param name="block_dim">wymiary bloku</param>
/// <param name="grid_dim">wymairy siatki</param>
/// <param name="size">wilekosc danych</param>
void CalculateGridBlockDimensions(dim3& block_dim, dim3& grid_dim, int size);
/// <summary>
/// Funkcja alokujaca tablice int2 na GPU pozwala na przekopiowanie danych z tablicy na CPU
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="tab">tablica</param>
/// <param name="count">liczba komorek</param>
/// <param name="size"></param>
/// <param name="lasttab">tablica na CPU</param>
void MallocCudaInt2(cudaError_t& cudaStatus, int2*& tab, int count, int size, int2* lasttab);
/// <summary>
/// Funkcja alokujaca tablice int na GPU pozwala na przekopiowanie danych z tablicy na CPU
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="tab">tablica</param>
/// <param name="count">liczba komorek</param>
/// <param name="size"></param>
/// <param name="lasttab">tablica na CPU</param>
void MallocCudaInt(cudaError_t& cudaStatus, int*& tab, int count, int size, int* lasttab);
/// <summary>
/// Funkcja alokujaca tablice float na GPU pozwala na przekopiowanie danych z tablicy na CPU
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="tab">tablica</param>
/// <param name="count">liczba komorek</param>
/// <param name="size"></param>
/// <param name="lasttab">tablica na CPU</param>
void MallocCudaFloat(cudaError_t& cudaStatus, float*& tab, int count, int size, float* lasttab);
/// <summary>
/// Funkcja kopiujaca dane z GPU na CPU
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="px">tablica GPU pozycji x</param>
/// <param name="py">tablica GPU pozycji y</param>
/// <param name="pvx">tablica GPU predkosci x</param>
/// <param name="pvy">tablica GPU predkosci y</param>
/// <param name="x">tablica CPU pozycji x</param>
/// <param name="y">tablica CPU pozycji y</param>
/// <param name="vx">tablica CPU predkosci x</param>
/// <param name="vy">tablica CPU predkosci y</param>
/// <param name="params">parametry symulacji</param>
void MemcpyCuda(cudaError_t& cudaStatus, float* px, float* py, float* pvx, float* pvy, float* x, float* y, float* vx, float* vy, Parameters_t& params);
/// <summary>
/// Funkcja sprawdzajaca poprawnosc wywolanie kernela oraz go synchornizujaca
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="str">teskt dodawany do wypisywanego bledu</param>
void CheckForErrorsAndSynchronize(cudaError_t& cudaStatus, const char* str);
/// <summary>
/// Funkcja sprawdzajaca kod bledu
/// </summary>
/// <param name="cudaStatus">kod bledu</param>
/// <param name="str">teskt dodawany do wypisywanego bledu</param>
void CheckForErrors(cudaError_t& cudaStatus, const char* str);
/// <summary>
/// Funckja zwalnia tablice alokowana na GPU
/// </summary>
void FreeCudaArray(void* pointer);

//Funkcje uruchamianie z GPU

/// <summary>
/// Funckja obliczajaca pozycje w tablicy zderzen na podstawie polozenia czasteczki
/// </summary>
/// <param name="x">polozenie "x"</param>
/// <param name="y">polozenie "y"</param>
/// <param name="params">parametry symulacji</param>
/// <returns>pozycja w tablicy (cell_id)</returns>
__device__ int CalculateCellId(float x, float y, Parameters_t& params);
/// <summary>
/// Funckja obliczajaca nowa predkosc czasteczki o numerze "i" po zderzeniu z casteczka o numerze "j"
/// </summary>
/// <param name="i"></param>
/// <param name="j"></param>
/// <param name="newvel">struktura zawierajaca tablice nowych predkosci</param>
/// <param name="part">struktura zawierajaca dane czasteczek</param>
/// <param name="params">struktura z parametrami symulacji</param>
__device__ void CalculateNewVelocity(int i, int j, NewVelocities_t& newvel, Particles_t& part, Parameters_t& params);


//Kernele

/// <summary>
/// Kernel obliczajacy wplyw grawitacji i przsuwajacy czasteczki na nowe pozycje na podstawie ich predkosci
/// </summary>
/// <param name="part">struktura zawierajaca dane czasteczek</param>
/// <param name="dt"></param>
/// <param name="cellsx">tablica pomocnicza do budowy tablicy/siatki zderzen</param>
/// <param name="cellsy">tablica pomocnicza do budowy tablicy/siatki zderzen</param>
/// <param name="params">parametry symulacji</param>
/// <returns></returns>
__global__ void gravity_and_damp_kernel(Particles_t part, float dt, int* cellsx, int* cellsy, Parameters_t params);

/// <summary>
/// Kernel wypelniajacy tablice pomocnicza wartosciami (-1,-1)
/// </summary>
/// <param name="startend">tablica pomocnicza</param>
/// <param name="params">parametry symulacji</param>
/// <returns></returns>
__global__ void fill_startend(int2* startend, Parameters_t params);

/// <summary>
/// Kernel wyszukujacy pozycje startowe i koncowe komorki "i" w tablicy cellsx i zapisujacy te informacje w tablicy startend
/// </summary>
/// <param name="cellsx"></param>
/// <param name="cellsy"></param>
/// <param name="startend"></param>
/// <param name="params">parametry symulacji</param>
/// <returns></returns>
__global__ void find_startend_kernel(int* cellsx, int* cellsy, int2* startend, Parameters_t params);

/// <summary>
/// Kerenel obliczaj¹cy zachodz¹ce kolizje i ich wplyw na czasteczki, dzieki zastosowaniu talbic pomocniczych w znaczny sposob
///		zmniejszony jest naklad obliczeniowy - sprawdzane sa tylko kolizje ktore maja szanse zajsc ( oczywiscie w przypadku grupowania sie
///		czasteczek np. na dnie okna wydajnosc znacznie sie zmniejsza, gdzyz dana czasteczka moze kolidowac prawie ze wszytkimi innymi)
/// </summary>
/// <param name="particles">struktura zawierajaca dane o czasteczkach</param>
/// <param name="cellsx">pomocnicza tablica zderzen</param>
/// <param name="cellsy">pomocnicza tablica zderzen</param>
/// <param name="startend">pomocnicza tablica zderzen</param>
/// <param name="newVelocities">struktura zawierajaca nowe predkosci</param>
/// <param name="params">parametry symulacji</param>
/// <returns></returns>
__global__ void collision_kernel(Particles_t particles, int* cellsx, int* cellsy, int2* startend, NewVelocities_t newVelocities, Parameters_t params);

/// <summary>
/// Kernel aktualizujacy wartosci predkosci czasteczek na nowo obliczone
/// </summary>
/// <param name="particles">struktura zawierajaca dane o czasteczkach</param>
/// <param name="newVelocities">struktura zawierajaca nowe predkosci</param>
/// <param name="params">parametry symulacji</param>
__global__ void calculate_new_velocity_kernel(Particles_t particles, NewVelocities_t newvel, Parameters_t params);