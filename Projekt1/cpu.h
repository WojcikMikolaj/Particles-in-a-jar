#pragma once
#include "kernel.cuh"
#include <algorithm>
///
/// 
/// Analogicznie jak w kernel.cuh
/// 
/// 
int CalculateCellId_CPU(float x, float y, Parameters_t& params);

void CalculateNewVelocity_CPU(int i, int j, NewVelocities_t& newvel, Particles_t& particles, Parameters_t& params);

void gravity_and_damp_cpu(int i, Particles_t particles, float dt, int* cellsx, int* cellsy, Parameters_t params);

void fill_startend_cpu(int i, int2* startend, Parameters_t params);

void find_startend_cpu(int i, int* cellsx, int* cellsy, int2* startend, Parameters_t params);

void collision_cpu(int i, Particles_t particles, int* cellsx, int* cellsy, int2* startend, NewVelocities_t newVelocities, Parameters_t params);

void calculate_new_velocity_cpu(int i, Particles_t particles, NewVelocities_t newvel, Parameters_t params);

float RunCpu(Particles_t& cpu_part, NewVelocities_t& cpu_newvel, float t, Parameters_t& params, Cells_t& cpu_cells, float& cpytime);