#include "Coordinates.h"

/**
Updates the realCoordinate vector with physiological coordinates in space.
Requires the index coordinates, the length of a single NVU in meters, and the
grid dimensions.
Currently the Z-dimension is not being used, so all coordinates are given a Z
value of 0.0.
*/
void calculateRealCoordinates(std::vector<int> indexCoordinate, double length,
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
bool isInNVUGrid(std::vector<double> realCoordinate,
		double length, 
		int xNumberNVUs, int yNumberNVUs, 
		int zNumberNVUs)
{
  int size = realCoordinate.size();
  double *rCoord = &realCoordinate[0];
  return isInNVUGrid(rCoord, size, length,
      xNumberNVUs, yNumberNVUs, zNumberNVUs);
}
bool isInNVUGrid(double *realCoordinate, int size, double length,
                 int xNumberNVUs, int yNumberNVUs, int zNumberNVUs)
{
  std::vector<int> indexCoordinate;
  calculateIndexCoordinates(realCoordinate, size, length, xNumberNVUs,
                            yNumberNVUs, zNumberNVUs, indexCoordinate);
  bool result = false;
  if (indexCoordinate[0] < xNumberNVUs && indexCoordinate[1] < yNumberNVUs &&
      indexCoordinate[2] < zNumberNVUs)
    result = true;
  return result;
}
