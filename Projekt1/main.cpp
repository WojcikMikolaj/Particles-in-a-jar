#include "kernel.cuh"
#include <time.h>
#include "main.h"
#include <iostream>
#include <fstream>
#include <string>
#include "cpu.h"


/// <summary>
/// Zmienne globalne, ich u¿ycie jest niestety wymuszone przez u¿ycie biblioteki GLUT
/// </summary>
Particles_t particles;
Particles_t dev_particles;
NewVelocities_t dev_newvel;
NewVelocities_t cpu_newvel;
Parameters_t params;
Cells_t dev_cells;
Cells_t cpu_cells;
float dt = 0.001f;
volatile float last;
volatile bool cpu = false;
volatile unsigned int fps = 0;
volatile unsigned int frames = 0;

/// <summary>
/// Funkcja oblicza wartoœæ FPS aminacji
/// </summary>
void ComputeFramerate()
{
	float now = std::clock();
	if (now - last >= CLOCKS_PER_SEC)
	{
		last = now;
		fps = frames;
		frames = 0;
	}
}

/// <summary>
/// Funkcja odpowiedzialna za rysowanie cz¹steczek
/// </summary>
void display()
{
	float cpytime = 0.0f;
	float cudatime = 0.0f;


	/// <summary>
	///Obliczamy pozycjê cz¹steczek w kolejnej klatce animacji za pomoc¹ GPU (RunCuda) lub na CPU (RunCpu),
	///  w zale¿noœci od wartoœci zmiennej "CPU" 
	/// </summary>
	if (!cpu)
	{
		cudatime = RunCuda(particles, dev_particles, dev_newvel, dt, params, dev_cells, cpytime);
	}
	else
	{
		cudatime = RunCpu(particles, cpu_newvel, dt, params, cpu_cells, cpytime);
	}


	/// <summary>
	/// Konstrukcja tytu³u okna na którym wyœwietlane s¹:
	///		- aktualna wartoœæ FPS
	///		- ile czasu zajê³o oblicznie nowej pozycji cz¹steczek w [ms] i czy by³o to wykonane na CPU czy GPU
	///		- ile czasu zajê³o kopiowanie danych z GPU do CPU po zakoñczeniu obliczeñ w [ms] 
	/// </summary>
	std::string title = "FPS: ";
	title += std::to_string(fps);
	title += ", ";
	if (cpu)
	{
		title += "CPU";
	}
	else
	{
		title += "CUDA";
	}
	title += " calculations time: ";
	title += std::to_string(cudatime);
	title += " ms, Copying time :";
	title += std::to_string(cpytime);
	title += " ms";
	glutSetWindowTitle(title.c_str());


	/// <summary>
	/// Rysowanie wszytkich cz¹steczek na ekranie w 3 kolorach
	/// </summary>
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0, 1.0, 0.0);
	glPointSize(params.particle_size);
	glBegin(GL_POINTS);
	for (int i = 0; i < particles.size / 3; i++)
	{
		glVertex2f(particles.x[i], particles.y[i]);
	}
	glEnd();
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_POINTS);
	for (int i = particles.size / 3; i < particles.size * (2.0 / 3); i++)
	{
		glVertex2f(particles.x[i], particles.y[i]);
	}
	glEnd();
	glColor3f(0.0, 1.0, 1.0);
	glBegin(GL_POINTS);
	for (int i = particles.size * (2.0 / 3); i < particles.size; i++)
	{
		glVertex2f(particles.x[i], particles.y[i]);
	}
	glEnd();
	glFlush();
}


/// <summary>
/// Funkcja wywo³uj¹ca funckjê odpowiedzialn¹ za rysowanie w sta³ych odstêpach czasu
/// </summary>
void timerfunc(int abc)
{
	frames++;
	//Obliczenie wartoœci FPS
	ComputeFramerate();
	//Ponowna rejestracja zdarzenia
	glutTimerFunc(8, timerfunc, 0);
	//Wywo³anie funckji odpowiedzialnej
	glutPostRedisplay();
}

//Przygotowanie okna animacji
void init(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}

//Funckja zwalniaj¹ca zasoby na GPU
void FreeAnimationMemory()
{
	if (!cpu)
	{
		FreeCudaArrays(dev_particles, dev_newvel, dev_cells);
	}
	else
	{
		delete[] particles.x;
		delete[] particles.y;
		delete[] cpu_newvel.nvx;
		delete[] cpu_newvel.nvy;
		delete[] cpu_cells.cells_x;
		delete[] cpu_cells.cells_y;
		delete[] cpu_cells.startend;
	}
}

/// <summary>
/// Alokacja pamiêci na dane oraz jej wype³nienie w przypadku wykonywania obliczeñ na CPU
/// </summary>
void ReallocParticlesForCpu()
{
	particles.size = params.particles_count;
	particles.mass = new float[particles.size];
	particles.x = new float[particles.size];
	particles.y = new float[particles.size];
	particles.vx = new float[particles.size];
	particles.vy = new float[particles.size];

	//Wypelnienie losowymi wartosciami poczatkowymi z odpowiednich przedzialow
	for (int i = 0; i < particles.size; i++)
	{
		particles.mass[i] = 1;
		particles.x[i] = rand() % params.width / (float)params.width;
		particles.y[i] = rand() % params.height / (float)params.height;
		particles.vx[i] = (rand() % 2 == 0 ? -1 : 1) * (rand() % MAX_START_VEL + rand() % 100 / (float)100);
		particles.vy[i] = (rand() % 2 == 0 ? -1 : 1) * (rand() % MAX_START_VEL + rand() % 100 / (float)100);
	}

	cpu_newvel.nvx = new float[particles.size];
	cpu_newvel.nvy = new float[particles.size];

	cpu_cells.cells_x = new int[particles.size];
	cpu_cells.cells_y = new int[particles.size];
	cpu_cells.startend = new int2[params.height * params.width];
}

//Realokacja pamieci na czasteczki w przypadku obliczen na GPU
void ReallocParticles()
{
	ReallocParticlesForCpu();
	PrepareCuda(particles, dev_particles, dev_newvel, params, dev_cells);
	delete[] particles.mass;
	delete[] particles.vx;
	delete[] particles.vy;
}

//Zaladowanie konfiguracji z pliku lub w przypadku braku takowej wykorzystanie parametrow z pliku "constants.h"
void LoadSimulationParams()
{
	params.inf_loop = 1;
	params.width = WIDTH;
	params.height = HEIGHT;
	params.eps = EPS;
	params.g_const = G_CONST;
	params.particles_count = PARTICLES_NUM;
	params.particle_size = PARTICLE_SIZE;
	params.collision_dumping_factor = 1;
	params.dumping_factor = DUMPINGFACTOR;
	std::ifstream file("params.txt");
	std::string str;
	if (file.is_open())
	{
		while (file >> str)
		{
			if (str._Equal("INF_LOOP"))
			{
				if (file >> str)
				{
					params.inf_loop = atoi(str.c_str());
				}
				continue;
			}
			if (str._Equal("WIDTH"))
			{
				if (file >> str)
				{
					params.width = atoi(str.c_str());
				}
				continue;
			}
			if (str._Equal("HEIGHT"))
			{
				if (file >> str)
				{
					params.height = atoi(str.c_str());
				}
				continue;
			}
			if (str._Equal("EPS"))
			{
				if (file >> str)
				{
					params.eps = atof(str.c_str());
				}
				continue;
			}
			if (str._Equal("G_CONST"))
			{
				if (file >> str)
				{
					params.g_const = atof(str.c_str());
				}
				continue;
			}
			if (str._Equal("PARTICLES_COUNT"))
			{
				if (file >> str)
				{
					params.particles_count = atoi(str.c_str());
				}
				continue;
			}
			if (str._Equal("PARTICLE_SIZE"))
			{
				if (file >> str)
				{
					params.particle_size = atoi(str.c_str());
				}
				continue;
			}
			if (str._Equal("DUMPING_FACTOR"))
			{
				if (file >> str)
				{
					params.dumping_factor = atof(str.c_str());
				}
				continue;
			}
			if (str._Equal("COLLISION_DUMPING_FACTOR"))
			{
				if (file >> str)
				{
					params.collision_dumping_factor = atof(str.c_str());
				}
				continue;
			}
		}
		file.close();
	}
}

//Przeladowanie animacji z realokacja danych
void ResetAnimation()
{
	if (cpu == false)
	{
		FreeAnimationMemory();
		ReallocParticles();
	}
	else
	{
		delete[] particles.x;
		delete[] particles.y;
		delete[] cpu_newvel.nvx;
		delete[] cpu_newvel.nvy;
		delete[] cpu_cells.cells_x;
		delete[] cpu_cells.cells_y;
		delete[] cpu_cells.startend;
		ReallocParticlesForCpu();
	}
}

/// <summary>
/// Funkcja wywolywana przy zmianie wielkosci okna
/// </summary>
void ChangeWindowSize(int width, int height)
{
	params.height = height;
	params.width = width;
	if (!cpu)
	{
		cudaError_t cudaStatus;
		FreeCudaArray(dev_cells.startend);
		MallocCudaInt2(cudaStatus, dev_cells.startend, params.width * params.height, sizeof(int2), nullptr);
	}
	else
	{
		delete[] cpu_cells.startend;
		cpu_cells.startend = new int2[params.height * params.width];
	}
	glViewport(0, 0, width, height);
}

/// <summary>
/// Funkcja odpowiedzialna za szczytywanie nacisniec klawiszy w trakcie dzialania programu
/// </summary>
void ProcessKeys(unsigned char key, int x, int y)
{
	//Zresetowanie animacji
	if (key == 'r')
	{
		ResetAnimation();
	}
	//Zmiana urzadzenia wykonujacego obliczenia z GPU na CPU i vice versa, wiaze sie to ze zresetowaniem animacji
	if (key == 'c')
	{
		if (cpu == false)
		{
			FreeAnimationMemory();
			delete[] particles.x;
			delete[] particles.y;
			ReallocParticlesForCpu();
		}
		else
		{
			delete[] cpu_newvel.nvx;
			delete[] cpu_newvel.nvy;
			delete[] cpu_cells.cells_x;
			delete[] cpu_cells.cells_y;
			delete[] cpu_cells.startend;
			ReallocParticles();
		}
		cpu = !cpu;
	}
	//Przelaczenie miedzy odbiciami od dolnej krawedzi okna, a "spadaniem w nieskonczonosc" - po wypadnieciu przez dolna krawedz
	//	czasteczka pojawia sie na gornej krawedzi
	if (key == 'i')
	{
		params.inf_loop = !params.inf_loop;
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 100
	if (key == '2')
	{
		params.particles_count = 100;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 1k
	if (key == '3')
	{
		params.particles_count = 1000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 10k
	if (key == '4')
	{
		params.particles_count = 10000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 100k
	if (key == '5')
	{
		params.particles_count = 100000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 200k
	if (key == '6')
	{
		params.particles_count = 200000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 300k
	if (key == '7')
	{
		params.particles_count = 300000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 500k
	if (key == '8')
	{
		params.particles_count = 500000;
		ResetAnimation();
	}
	//Zmiana liczby czasteczek obecnych w animacji na: 1m
	if (key == '9')
	{
		params.particles_count = 1000000;
		ResetAnimation();
	}
}

int main(int argc, char** argv)
{
	//Zaladowanie konfiguracji
	LoadSimulationParams();


	//Wybranie GPU
	bool retflag;
	int retval = SetCudaDevice(retflag);
	if (retflag)
		return retval;


	//Przygotowanie GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(params.width, params.height);
	glutInitWindowPosition(0, 0);


	//inicjalizacja struktury opisujacej czasteczki oraz jej wypelnienie
	particles.size = params.particles_count;
	particles.mass = new float[particles.size];
	particles.x = new float[particles.size];
	particles.y = new float[particles.size];
	particles.vx = new float[particles.size];
	particles.vy = new float[particles.size];
	srand((unsigned)time(NULL));
	for (int i = 0; i < particles.size; i++)
	{
		particles.mass[i] = 1;
		particles.x[i] = rand() % params.width / (float)params.width;
		particles.y[i] = rand() % params.height / (float)params.height;
		particles.vx[i] = (rand() % 2 == 0 ? -1 : 1) * (rand() % MAX_START_VEL + rand() % 100 / (float)100);
		particles.vy[i] = (rand() % 2 == 0 ? -1 : 1) * (rand() % MAX_START_VEL + rand() % 100 / (float)100);
	}

	//Przygotowanie odpowiadajacej struktury na GPU i przekopiowanie do niej danych
	PrepareCuda(particles, dev_particles, dev_newvel, params, dev_cells);
	//usuniecie zbednych tablic
	delete[] particles.mass;
	delete[] particles.vx;
	delete[] particles.vy;


	//Stworzenie okna aplikacji
	glutCreateWindow("A Simple OpenGL Windows Application with GLUT");

	//uruchomienie timera, wykorzystywanego do licznia wartosci FPS animacji
	last = std::clock();

	//inicjalizacja GLUT
	init();

	//Ustalenie funkcji rysujacej
	glutDisplayFunc(display);

	//rejestracja zdarzenia do wywolania za 8 ms
	glutTimerFunc(8, timerfunc, 0);

	//rejestracja funckji zwalniajacej pamiec w przypadku wyjscia z aplikacji
	atexit(FreeAnimationMemory);
	//rejestracja funckji obslugujacej wcisniecia klawiszy
	glutKeyboardFunc(ProcessKeys);
	//rejestracja zdarzenia zmieny rozmiary okno
	glutReshapeFunc(ChangeWindowSize);
	//wywolanie glownej petli aplikacji
	glutMainLoop();

	return 0;
}