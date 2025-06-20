// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
// Created by Heraldo Memelli
// summer 2012

#ifndef NEUROGENSEGMENT_H
#define NEUROGENSEGMENT_H

#include <iostream>
#include <string>
#include <math.h>
#include <map>
#include "ShallowArray.h"

#include "NeurogenSegment.h"
#include "NeurogenParams.h"

#include <cmath>
#include "rndm.h"

class NeurogenBranch;
class BoundingVolume;
class BoundingSurfaceMesh;

class NeurogenSegment {
 public:
  NeurogenSegment();
  NeurogenSegment(NeurogenSegment*);
  void set(const NeurogenSegment& seg);
  void set(NeurogenParams* params);
  void set(int id, int type, double x, double y, 
	   double z, double r, int p, NeurogenParams* params);
  void set(int type, double x, double y, 
	   double z, double r, NeurogenParams* params);
  virtual ~NeurogenSegment();
  std::string outputLine();
  std::string output();
  void setParent(NeurogenSegment* _parent);
  void grow();
  int getID();
  int getType();
  int getParent();
  double getX();
  double getY();
  double getZ();
  double getRadius();
  double getLength();
  bool touches(NeurogenSegment* other, double tolerance);
  bool equals(NeurogenSegment* other);
  double getAngle(NeurogenSegment*);
  double getDistance(NeurogenSegment*);
  double getDistance(ShallowArray<double>& coords);
  double getAngleFromParent();
  //double getRand();
  NeurogenBranch* getBranch();
  NeurogenSegment* getParentSeg();
  NeurogenParams* getParams();
  void setID(int);
  void setType(int);
  void setParentID(int);
  void setRadius(double _r);
  void setX(double);
  void setY(double);
  void setZ(double);
  void setStartingCoords();
  void setBranch(NeurogenBranch*);
  void reset();
  void resetBias();
  void resampleGaussian(NeurogenSegment*, double sd);
  void resampleGaussian(NeurogenSegment*);
  void resampleAfterForces();
  void resampleAfterForces(double sd);
  void rotateDaughters(NeurogenSegment* child1, NeurogenSegment* child2);
  void rotateDaughters2D(NeurogenSegment* child1, NeurogenSegment* child2);
  static void cross(double vec1[3], double vec2[3], double* vec3);
  static double dot(double vec1[3], double vec2[3]);
  void rotate2D(double angle);
  void setLength(double length);
  double getSideArea();
  double getVolume();
  void growSameDirectionAsParent();
  bool intersects(NeurogenSegment* other);
  void homotypicRepulsion(ShallowArray<NeurogenSegment*>& otherSegs);
  void somaRepulsion(NeurogenSegment* soma_p);
  void tissueBoundaryRepulsion(std::map<std::string, BoundingSurfaceMesh*>& boundingSurfaceMap);
  void waypointAttraction();
  void forwardBias();
  double getBiasX() {return biasX;}
  double getBiasY() {return biasY;}
  double getBiasZ() {return biasZ;}
  NeurogenSegment& operator=(const NeurogenSegment &s);
  void setNeuriteOrigin(bool isOrigin) {neuriteOrigin=isOrigin;}
  bool isNeuriteOrigin() {return neuriteOrigin;}

 private:
  int ID;
  int Type;
  double X;
  double Y;
  double Z;
  double Radius;
  int Parent;

  //auxiliary members
  NeurogenParams* params_p;
  NeurogenSegment* parent_p;
  NeurogenBranch* branch_p;
  double biasX;
  double biasY;
  double biasZ;
  bool neuriteOrigin;
};

#endif /* SEGMENT_H_ */
