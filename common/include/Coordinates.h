#ifndef _COORDINATES_H
#define _COORDINATES_H

#include <math.h>
#include <vector>


void calculateRealCoordinatesNVU(std::vector<int> indexCoordinate, double length, int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs, std::vector<double> &realCoordinate);
void _calculateRealCoordinates(std::vector<int> indexCoordinate, double length, int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs, std::vector<double> &realCoordinate);
void _calculateRealCoordinatesCenter(
		std::vector<int> indexCoordinate, double length, 
		int xNumberNVUs, int yNumberNVUs, int zNumberNVUs, 
		std::vector<double> &realCoordinate);
void calculateRealCoordinatesNTS(
		std::vector<int> indexCoordinate, double length, 
		int xNumberNVUs, int yNumberNVUs, int zNumberNVUs, 
		std::vector<double> &realCoordinate);
void calculateRealCoordinatesCenterNTS(
		std::vector<int> indexCoordinate, double length, 
		int xNumberNVUs, int yNumberNVUs, int zNumberNVUs, 
		std::vector<double> &realCoordinate);

void calculateIndexCoordinates(std::vector<double> realCoordinate, double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs, std::vector<int> &indexCoordinate);
void calculateIndexCoordinates(double* realCoordinate, int size,
		double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs, std::vector<int> &indexCoordinate);

/*
 * realCoordinate: in micro-meter unit
 * length = in micro-meter unit
 */
bool isInNVUGrid(std::vector<double> realCoordinate,
		double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs);
bool isInNVUGrid(double* realCoordinate, int size,
		double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs);

#endif
