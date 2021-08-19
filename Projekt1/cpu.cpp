#include "cpu.h"
///
///
///Analogicznie jak w kernel.cu
///  
/// 
int CalculateCellId_CPU(float x, float y, Parameters_t& params)
{
	int id = ((y * params.width * params.height) / (float)params.particle_size + x * params.width) / (float)params.particle_size;
	return id;
}

void CalculateNewVelocity_CPU(int i, int j, NewVelocities_t& newvel, Particles_t& particles, Parameters_t& params)
{
	if (i == j)
	{
		return;
	}
	float masspart = 2 * particles.mass[j] / (particles.mass[i] + particles.mass[j]);
	float2 v2 = particles.GetVelocity(j);
	//float2 v1 = particles.GetVelocity(i);
	float2 v1 = make_float2(newvel.nvx[i], newvel.nvy[i]);
	float2 x2 = particles.GetPosition(j);
	float2 x1 = particles.GetPosition(i);
	float2 dx = x1 - x2;
	float2 dv = v1 - v2;
	float ldx = length(dx);
	if (fabsf(dx.x) < params.particle_size && fabsf(dx.y) < params.particle_size)
	{
		float2 newV = masspart * dot(dx, dv) / (ldx * ldx) * dx;
		newvel.nvx[i] = newvel.nvx[i] - newV.x;
		newvel.nvy[i] = newvel.nvy[i] - newV.y;
	}
	return;
}

void gravity_and_damp_cpu(int i, Particles_t particles, float dt, int* cellsx, int* cellsy, Parameters_t params)
{
	if (i >= params.particles_count)
		return;

	particles.vy[i] += params.g_const * dt;

	particles.x[i] += particles.vx[i] * dt;
	particles.y[i] += particles.vy[i] * dt;

	if (particles.y[i] <= 0 || particles.y[i] >= 1.0)
	{
		particles.vy[i] *= -params.dumping_factor;

		if (particles.y[i] < 0)
		{
			
			particles.y[i] = params.inf_loop;
		}
		else if (particles.y[i] > 1)
		{
			
			particles.y[i] = 1;
		}
	}

	if (particles.x[i] <= 0 || particles.x[i] >= 1.0)
	{
		particles.vx[i] *= -params.dumping_factor;

		if (particles.x[i] < 0)
		{

			particles.x[i] = 0;
		}
		else if (particles.x[i] > 1)
		{

			particles.x[i] = 1;
		}
	}

	cellsx[i] = CalculateCellId_CPU(particles.x[i], particles.y[i], params);
	cellsy[i] = i;
}

void fill_startend_cpu(int i, int2* startend, Parameters_t params)
{
	if (i < params.height * params.width)
	{
		startend[i].x = -1;
		startend[i].y = -1;
	}
}

void find_startend_cpu(int i, int* cellsx, int* cellsy, int2* startend, Parameters_t params)
{
	if (i > 0 && i < params.particles_count)
	{
		if (cellsx[i - 1] != cellsx[i])
		{
			startend[cellsx[i]].x = i;
			startend[cellsx[i - 1]].y = i;
		}
	}
	if (i == 0)
	{
		startend[cellsx[0]].x = 0;
		startend[cellsx[params.particles_count - 1]].y = params.particles_count;
	}
}

void collision_cpu(int i, Particles_t particles, int* cellsx, int* cellsy, int2* startend, NewVelocities_t newVelocities, Parameters_t params)
{

	if (i >= params.particles_count)
		return;

	if (i < params.particles_count)
	{
		//calculate cellid of a paticle
		int cellid = CalculateCellId_CPU(particles.x[i], particles.y[i], params);
		//forces
		newVelocities.nvx[i] = particles.vx[i];
		newVelocities.nvy[i] = particles.vy[i];
		//CleanForce(force);

		//cells to check
		//-------------
		//| 0 | 1 | 2 | 
		//-------------
		//| 3 | 4 | 5 | 
		//-------------
		//| 6 | 7 | 8 | 
		//-------------

		//cell 4 
		//position (x,y)
		if (cellid>=0&&startend[cellid].x != -1 && startend[cellid].y != -1)
		{
			for (int j = startend[cellid].x; j < startend[cellid].y; j++)
			{
				if (cellsy[j] != i && cellsx[j] == cellid)
				{
					CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
				}
			}
		}

		//cell 0
		//position (x-1,y+1)
		if (cellid - 1 + params.width / params.particle_size < params.MaxCellId()&& cellid - 1 + params.width / params.particle_size>=0)
		{
			if (startend[cellid - 1 + params.width / params.particle_size].x != -1 && startend[cellid - 1 + params.width / params.particle_size].y != -1)
			{
				for (int j = startend[cellid - 1 + params.width / params.particle_size].x; j < startend[cellid - 1 + params.width / params.particle_size].y; j++)
				{
					if (cellsy[j] != i)
					{
						CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
					}
				}
			}
		}

		////cell 1
		////position (x,y+1)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 2
		////position (x+1,y+1)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 3
		////position (x-1,y)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		////cell 5
		////position (x+1,y)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		//cell 6
		//position (x-1,y-1)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
		//cell 7
		//position (x,y-1)
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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}

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
							CalculateNewVelocity_CPU(i, cellsy[j], newVelocities, particles, params);
						}
					}
				}
			}
		}
	}
}

void calculate_new_velocity_cpu(int i, Particles_t particles, NewVelocities_t newvel, Parameters_t params)
{

	if (i < params.particles_count)
	{
		if (fabsf(newvel.nvx[i]) > EPS)
		{
			particles.vx[i] = params.collision_dumping_factor * newvel.nvx[i];
		}
		if (fabsf(newvel.nvy[i]) > EPS)
		{
			particles.vy[i] = newvel.nvy[i];
		}
	}
}

float RunCpu(Particles_t& cpu_part, NewVelocities_t& cpu_newvel, float t, Parameters_t& params, Cells_t& cpu_cells, float& cpytime)
{
	float elapsed = 0;
	std::clock_t c_start = std::clock();
	cudaEvent_t start, stop;
	cudaError_t cudaStatus;

	// Setup grid & block dims
	dim3 block_dim, grid_dim;
	// Move using gravity and velocity from previous iteration, 
	// calculate particle position (cell) after completed move
	for (int i = 0; i < params.particles_count; i++)
	{
		gravity_and_damp_cpu(i, cpu_part, t, cpu_cells.cells_x, cpu_cells.cells_y, params);
	}

	int2* tab = new int2[params.particles_count];
	for (int i = 0; i < params.particles_count; i++)
	{
		tab[i].x = cpu_cells.cells_x[i];
		tab[i].y = cpu_cells.cells_y[i];
	}
	std::sort(tab, tab + params.particles_count, [](int2 x, int2 y) {return x.x < y.x; });
	for (int i = 0; i < params.particles_count; i++)
	{
		cpu_cells.cells_x[i] = tab[i].x;
		cpu_cells.cells_y[i] = tab[i].y;
	}
	delete[] tab;
	//fill indicies table with (-1,-1)
	for (int i = 0; i < params.height * params.width; i++)
	{
		fill_startend_cpu(i, cpu_cells.startend, params);
	}
	//calculate start/end indicies of each cell
	for (int i = 0; i < params.particles_count; i++)
	{
		find_startend_cpu(i, cpu_cells.cells_x, cpu_cells.cells_y, cpu_cells.startend, params);
	}


	//Process collisions
	for (int i = 0; i < params.particles_count; i++)
	{
		collision_cpu(i, cpu_part, cpu_cells.cells_x, cpu_cells.cells_y, cpu_cells.startend, cpu_newvel, params);
	}
	//Calculate velocity based on collision forces
	for (int i = 0; i < params.particles_count; i++)
	{
		calculate_new_velocity_cpu(i, cpu_part, cpu_newvel, params);
	}


	std::clock_t c_end = std::clock();
	cpytime = 0;
	elapsed = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

	return elapsed;
}