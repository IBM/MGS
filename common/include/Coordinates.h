/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================
(C) Copyright University of Canterbury 2017-2018. All rights reserved.

================================================================
*/
#ifndef _COORDINATES_H
#define _COORDINATES_H

#include <math.h>
#include <vector>

/* IMPORTANT: as long as 
 * realCordinate and length in the same unit
 * They should be ok
 */

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
