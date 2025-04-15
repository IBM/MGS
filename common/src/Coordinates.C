/*
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
// (C) Copyright University of Canterbury 2017-2018. All rights reserved.
//
// =============================================================================
#include <algorithm>
#include "Coordinates.h"

/**
Updates the realCoordinate vector with physiological coordinates in space.
Requires the index coordinates, the length of a single NVU in meters, and the
grid dimensions.
Currently the Z-dimension is not being used, so all coordinates are given a Z
value of 0.0.
*/
void calculateRealCoordinatesNVU(std::vector<int> indexCoordinate, double length,
                              int xNumberNVUs, int yNumberNVUs, int zNumberNVUs,
                              std::vector<double> &realCoordinate)
{
  double x, y, z;

  x = indexCoordinate[0] * 2 * length - ((xNumberNVUs - 1) * length);
  y = indexCoordinate[1] * 2 * length - ((yNumberNVUs - 1) * length);

  realCoordinate.push_back(x);
  realCoordinate.push_back(y);

  // No real implementation for Z axis coordinates/3D simulations YET.
  if (indexCoordinate.size() == 3)
  {
    z = 0.0;
    realCoordinate.push_back(z);
  }
}
void _calculateRealCoordinates(std::vector<int> indexCoordinate, double length,
                              int xNumberNVUs, int yNumberNVUs, int zNumberNVUs,
                              std::vector<double> &realCoordinate)
{
  double x, y, z;

  x = indexCoordinate[0] * 2 * length - ((xNumberNVUs - 1) * length);
  y = indexCoordinate[1] * 2 * length - ((yNumberNVUs - 1) * length);
  z = indexCoordinate[2] * 2 * length - ((zNumberNVUs - 1) * length);

  realCoordinate.push_back(x);
  realCoordinate.push_back(y);
  realCoordinate.push_back(z);
}

/*
 -----------------------------------
 |                                 |
 |                                 |
 |              *                  |
 |                                 |
 |_________________________________|
 return coord of (*)
 */
void _calculateRealCoordinatesCenter(std::vector<int> indexCoordinate, double length,
                              int xNumberNVUs, int yNumberNVUs, int zNumberNVUs,
                              std::vector<double> &realCoordinate)
{
  double x, y, z;

  x = indexCoordinate[0] * 2 * length - ((xNumberNVUs - 1) * length) ; //+ length/2;
  y = indexCoordinate[1] * 2 * length - ((yNumberNVUs - 1) * length) ; //+ length/2;
  z = indexCoordinate[2] * 2 * length - ((zNumberNVUs - 1) * length) ; //+ length/2;

  realCoordinate.push_back(x);
  realCoordinate.push_back(y);
  realCoordinate.push_back(z);
}

/* 
 * Information is passed in as (Z,Y,X) and returned as (X,Y,Z)
 */
/* return lower-left corner */
void calculateRealCoordinatesNTS(std::vector<int> indexCoordinate, double length,
                              int xNumberNVUs, int yNumberNVUs, int zNumberNVUs,
                              std::vector<double> &realCoordinate)
{
  _calculateRealCoordinates(indexCoordinate, length, xNumberNVUs, yNumberNVUs, zNumberNVUs, realCoordinate);// grid's dimension is returned as (Z,Y,X)
  std::reverse(realCoordinate.begin(),realCoordinate.end()); //to map to (X,Y,Z) the same as NTS
}

/* return center-point coordinate */
void calculateRealCoordinatesCenterNTS(std::vector<int> indexCoordinate, double length,
                              int xNumberNVUs, int yNumberNVUs, int zNumberNVUs,
                              std::vector<double> &realCoordinate)
{
  _calculateRealCoordinatesCenter(indexCoordinate, length, xNumberNVUs, yNumberNVUs, zNumberNVUs, realCoordinate);// grid's dimension is returned as (Z,Y,X)
  std::reverse(realCoordinate.begin(),realCoordinate.end()); //to map to (X,Y,Z) the same as NTS
}

/**
Updates the indexCoordinate vector with index coordinates within the MGS grid.
Requires the physiological coordinates in space, the length of a single NVU in
meters, and the grid dimensions.
Currently the Z-dimension is not being used, so all coordinates are given a Z
value of 0.
*/
void calculateIndexCoordinates(std::vector<double> realCoordinate,
                               double length, int xNumberNVUs, int yNumberNVUs,
                               int zNumberNVUs,
                               std::vector<int> &indexCoordinate)
{
  double *rCoord = &realCoordinate[0];
  int size = realCoordinate.size();
  calculateIndexCoordinates(rCoord, size, length, xNumberNVUs, yNumberNVUs,
                            zNumberNVUs, indexCoordinate);
}
void calculateIndexCoordinates(double *realCoordinate, int size, double length,
                               int xNumberNVUs, int yNumberNVUs,
                               int zNumberNVUs,
                               std::vector<int> &indexCoordinate)
{
  int x, y, z;

  x = (int)round((realCoordinate[0] + ((xNumberNVUs - 1) * length)) /
                 (2 * length));
  y = (int)round((realCoordinate[1] + ((yNumberNVUs - 1) * length)) /
                 (2 * length));

  indexCoordinate.push_back(x);
  indexCoordinate.push_back(y);

  // No real implementation for Z axis coordinates/3D simulations YET.
  if (size == 3)
  {
    z = 0;
    indexCoordinate.push_back(z);
  }
}
/* 
 * realCoordinate using NTS's coord convention: [0] ~ X, [1] ~ Y, [2] ~ Z-axis 
 */
bool isInNVUGrid(std::vector<double> realCoordinate,
		double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs)
{
  int ndims = realCoordinate.size();
  double *rCoord = &realCoordinate[0];
  return isInNVUGrid(rCoord, ndims, length,
      xNumberNVUs, yNumberNVUs, zNumberNVUs);
}
bool isInNVUGrid(double *realCoordinate, int size, double length,
                 int xNumberNVUs, int yNumberNVUs, int zNumberNVUs)
{
  std::vector<int> indexCoordinate;
  float lx = - length * 2 * (xNumberNVUs/2);
  float ux = length * 2 * (xNumberNVUs/2);
  float ly = - length * 2 * (yNumberNVUs/2);
  float uy = length * 2 * (yNumberNVUs/2);
  float lz = - length * 2 * (zNumberNVUs/2);
  float uz = length * 2 * (zNumberNVUs/2);
  bool result = false;
  if (
      lx <= realCoordinate[0] and  realCoordinate[0] <= ux and 
      ly <= realCoordinate[1] and  realCoordinate[1] <= uy and 
      lz <= realCoordinate[2] and  realCoordinate[2] <= uz 
     )
    result = true;
  return result;
}
