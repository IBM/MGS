// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

/*
 * NeurogenSegment.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: heraldo
 */

#include <sstream>
#include <iostream>
#include <string>

#include "NeurogenSegment.h"
#include "NeurogenBranch.h"
#include "BoundingVolume.h"
#include "BoundingSurfaceMesh.h"

#define SMALL_NUM 0.0000000001
#define SMALL_SD 0.0001

// Default Constructor
NeurogenSegment::NeurogenSegment()
  : ID(0),
    Type(0),
    X(0),
    Y(0),
    Z(0),
    Radius(0),
    Parent(-1),
    params_p(0),
    parent_p(0),
    branch_p(0),
    biasX(0),
    biasY(0),
    biasZ(0),
    neuriteOrigin(false)
{
}

NeurogenSegment::NeurogenSegment(NeurogenSegment* seg) 
  : ID(seg->ID),
    Type(seg->Type),
    X(seg->X),
    Y(seg->Y),
    Z(seg->Z),
    Radius(seg->Radius),
    Parent(seg->Parent),
    params_p(seg->params_p),
    parent_p(seg->parent_p),
    branch_p(seg->branch_p),
    biasX(seg->biasX),
    biasY(seg->biasY),
    biasZ(seg->biasZ),
    neuriteOrigin(false)
{
}

void NeurogenSegment::set(const NeurogenSegment& seg)
{
  ID=seg.ID;
  Type=seg.Type; 
  X=seg.X; 
  Y=seg.Y; 
  Z=seg.Z; 
  Radius=seg.Radius; 
  Parent=seg.Parent;
  params_p=seg.params_p;
  parent_p=seg.parent_p;
  branch_p=seg.branch_p;
  biasX=seg.biasX;
  biasY=seg.biasY;
  biasZ=seg.biasZ;
}

void NeurogenSegment::set(NeurogenParams* params) 
{
  ID = 1;
  Type = 1;
  X = 0;
  Y = 0;
  Z = 0;
  Radius = 5;
  Parent = -1;
  params_p = params;
  parent_p = 0;
  branch_p = 0;
  biasX = 0;
  biasY = 0;
  biasZ = 0;
}

void NeurogenSegment::set(int id, int type, double x, double y, double z, double r, int p, NeurogenParams* params)
{
  ID = id;
  Type = type;
  X = x;
  Y = y;
  Z = z;
  Radius = r;
  Parent = p;
  params_p = params;
  parent_p = 0;
  branch_p = 0;
  biasX = 0;
  biasY = 0;
  biasZ = 0;
}

void NeurogenSegment::set(int type, double x, double y, double z, double r, NeurogenParams* params)
{
  ID = 1;
  Type = type;
  X = x;
  Y = y;
  Z = z;
  Radius = r;
  Parent = -1;
  params_p = params;
  parent_p = 0;
  branch_p = 0;
  biasX = 0;
  biasY = 0;
  biasZ = 0;
}

double NeurogenSegment::getX()
{
  return X;
}

double NeurogenSegment::getY()
{
  return Y;
}

double NeurogenSegment::getZ()
{
  return Z;
}

double NeurogenSegment::getRadius()
{
  return Radius;
}

int NeurogenSegment::getID()
{
  return ID;
}

int NeurogenSegment::getType()
{
  return Type;
}

int NeurogenSegment::getParent()
{
  return Parent;
}

NeurogenBranch* NeurogenSegment::getBranch()
{
  return branch_p;
}

NeurogenSegment* NeurogenSegment::getParentSeg()
{
  return parent_p;
}

NeurogenParams* NeurogenSegment::getParams()
{
  return params_p;
}

void NeurogenSegment::setParent(NeurogenSegment * _parent)
{
  parent_p = _parent;
}

bool NeurogenSegment::intersects(NeurogenSegment* other)
{
  bool rval=false;
  double x2 = X;
  double x1 = 0;
  if(parent_p)
    x1 = parent_p->getX();

  double y2 = Y;
  double y1 = 0;
  if(parent_p)
    y1 = parent_p->getY();

  double x4 = other->getX();
  double x3 = 0;
  if(other->parent_p)
    x3 = other->parent_p->getX();

  double y4 = other->getY();
  double y3 = 0;
  if(other->parent_p)
    y3 = other->parent_p->getY();


  double ua_num = (x4-x3)*(y1-y3) - (y4-y3)*(x1-x3);
  double den = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1);

  double ub_num = (x2-x1)*(y1-y3) - (y2-y1)*(x1-x3);

  double ua = ua_num/den;
  double ub = ub_num/den;

  //std::cout << "ua_num " << ua_num << " den " << den << " ua " << ua << " ub_num " << ub_num << " ub " << ub << std::endl;
  //std::cout << "x2 " << x2 << " x1 " << x1 << " y1 " << y1 << " y3 " << y3 << " y2 " << y2 << std::endl;
  if ((ua<=1 && ua>=0) && (ub<=1 && ub>=0)) {
  //std::cout << "NeurogenSegment " << this->parent_p->outputLine() << "\t" << this->outputLine() << std::endl;
  //std::cout << "intersects with: " << other->parent_p->outputLine() << "\t" <<  other->outputLine() << std::endl;
    rval=true;
  }
  return rval;
}

void NeurogenSegment::setBranch(NeurogenBranch * _b)
{
  branch_p = _b;
}

std::string NeurogenSegment::outputLine()
{
  std::stringstream line;

  line << ID << " " << Type  << " " << X << " " << Y << " " << Z << " " << Radius << " " << Parent << " ";
  std::string mystr = line.str();
  return mystr;
}

std::string NeurogenSegment::output()
{
  std::stringstream line;

  line  << Type  << " " << X << " " << Y << " " << Z << " " << Radius << " " << Parent << " ";
  std::string mystr = line.str();
  return mystr;
}

bool NeurogenSegment::touches(NeurogenSegment* other, double tolerance)
{
  bool rval=false;
  double sumRadii = Radius + other->getRadius() + tolerance;
  if (getDistance(other) < sumRadii)
    rval=true;
  return rval;
}

void NeurogenSegment::rotateDaughters2D(NeurogenSegment* child1, NeurogenSegment* child2)
{
  assert(this==child1->getParentSeg());
  assert(this==child2->getParentSeg());
  double minangle = params_p->minBifurcationAngle/2.0;
  double maxangle = params_p->maxBifurcationAngle/2.0;
  double theta = minangle + (maxangle - minangle)*drandom(0, 1.0, params_p->_rng);
  child1->rotate2D(theta);
  child2->rotate2D(-theta);
}


void NeurogenSegment::rotate2D(double _angle)
{
  double angle = _angle;
  double deltaX = 0;
  double deltaY = 0;

  deltaX = X - parent_p->getX();
  deltaY = Y - parent_p->getY();

  double newX = deltaX * cos(angle) - deltaY * sin(angle); // now x is something different than original vector x
  double newY = deltaX * sin(angle) + deltaY * cos(angle);

  X = parent_p->getX() + newX;
  Y = parent_p->getY() + newY;
}

void NeurogenSegment::rotateDaughters(NeurogenSegment* child1, NeurogenSegment* child2)
{
  double lFront1=child1->getLength();
  double lFront2=child2->getLength();

  child1->setLength(1.0);
  child2->setLength(1.0);

  double minangle = params_p->minBifurcationAngle/2.0;
  double maxangle = params_p->maxBifurcationAngle/2.0;

  double d0[3] = {(child1->getX() - X), (child1->getY() - Y), (child1->getZ() - Z)};
  double orig[3] = {0, 0, 1};
  double theta = minangle + (maxangle - minangle)*drandom(0, 1.0, params_p->_rng);
 
  double z = cos(theta);

  double t = 2*M_PI*drandom(0, 1.0, params_p->_rng);

  double r = sqrt(1 - z * z );

  double x1 = r * cos(t);
  double y1 = r * sin(t);
  double x2 = -x1;
  double y2 = -y1;

  double k[3];

  cross(d0, orig, k);
  //std::cout << k[0] << std::endl;


  double costheta = dot(d0, orig);

  double R[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0} };

  double fac = (1 - costheta) / dot(k, k);

  for (int i =0; i<3; i++) {
    R[i][i] = R[i][i] + costheta;
    for (int j=0; j<3; j++) {
      R[i][j] = R[i][j] + k[i]*k[j] * fac;
    }
  }
  
  double d1[3], d2[3];
  d1[0] = x1;
  d1[1] = y1;
  d1[2] = z;

  d2[0] = x2;
  d2[1] = y2;
  d2[2] = z;

  double d1_prime[3], d2_prime[3];
  for (int i=0; i<3; i++) {
    d1_prime[i] = 0;
    d2_prime[i] = 0;
  }
  
  for (int i=0;i<3; i++) {
    for (int j=0; j<3; j++) {
      d1_prime[i] = d1_prime[i] + d1[j]*R[j][i];
      d2_prime[i] = d2_prime[i] + d2[j]*R[j][i];
    }
  }
  //d1_prime = d1 * R;
  
  child1->setX(d1_prime[0] + X) ;
  child1->setY(d1_prime[1] + Y) ;
  child1->setZ(d1_prime[2] + Z) ;
  child2->setX(d2_prime[0] + X) ;
  child2->setY(d2_prime[1] + Y) ;
  child2->setZ(d2_prime[2] + Z) ;

  child1->setLength(lFront1);
  child2->setLength(lFront2);
}

void NeurogenSegment::cross(double vec1[3], double vec2[3], double* vec3)
{
  vec3[0] = (vec1[1] * vec2[2]) - (vec2[1]*vec1[2]);
  vec3[1] = -(vec1[0]*vec2[2])+(vec2[0]*vec1[2]);
  vec3[2] = (vec1[0]*vec2[1])-(vec1[1]*vec2[0]);
}

double NeurogenSegment::dot(double vec1[3], double vec2[3])
{
  double dotproduct;
  dotproduct = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
  return dotproduct;

}

bool NeurogenSegment::equals(NeurogenSegment* other)
{
  return (this==other);
}

void NeurogenSegment::setStartingCoords()
{
  setX(params_p->startX);
  setY(params_p->startY);
  setZ(params_p->startZ);
}

double NeurogenSegment::getAngleFromParent()
{
  double angle = 0;
  double dirX = X - parent_p->getX();
  double dirY = Y - parent_p->getY();
  double dirZ = Z - parent_p->getZ();

  double magnitude = this->getLength();

  double dirXp = parent_p->getX() - parent_p->parent_p->getX();
  double dirYp = parent_p->getY() - parent_p->parent_p->getY();
  double dirZp = parent_p->getZ() - parent_p->parent_p->getZ();

  double mag_Parent = parent_p->getLength();

  double val = (dirX*dirXp + dirY*dirYp + dirZ*dirZp)/(magnitude * mag_Parent);

  angle = acos(val);

  double angleInDeg = angle * 180/M_PI;

  return angleInDeg;
}

double NeurogenSegment::getAngle(NeurogenSegment* otherSeg_p)
{
  double angle = 0;
  double dirX = X - parent_p->getX();
  double dirY = Y - parent_p->getY();
  double dirZ = Z - parent_p->getZ();

  double magnitude = this->getLength();

  double dirXp = otherSeg_p->getX() - otherSeg_p->parent_p->getX();
  double dirYp = otherSeg_p->getY() - otherSeg_p->parent_p->getY();
  double dirZp = otherSeg_p->getZ() - otherSeg_p->parent_p->getZ();

  double mag_Other = otherSeg_p->getLength();

  double val = (dirX*dirXp + dirY*dirYp + dirZ*dirZp)/(magnitude * mag_Other);

  angle = acos(val);

  return angle;
}

double NeurogenSegment::getDistance(NeurogenSegment* otherSeg_p)
{
  double distance = 0;
  double deltaX = X - otherSeg_p->getX();
  double deltaY = Y - otherSeg_p->getY();
  double deltaZ = Z - otherSeg_p->getZ();

  distance = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);

  return distance;
}

double NeurogenSegment::getDistance(ShallowArray<double>& coords)
{
  double distance = 0;
  double deltaX = X - coords[0];
  double deltaY = Y - coords[1];
  double deltaZ = Z - coords[2];

  distance = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);

  return distance;
}

double NeurogenSegment::getLength()
{
  double length = sqrt((X-parent_p->getX())*(X-parent_p->getX()) + (Y-parent_p->getY())*(Y-parent_p->getY()) + (Z-parent_p->getZ())*(Z-parent_p->getZ()));
  return length;
}

void NeurogenSegment::setRadius(double _r)
{
  Radius = _r;

}

void NeurogenSegment::setX(double _x)
{
  X = _x;
}

void NeurogenSegment::setY(double _y)
{
  Y = _y;
}

void NeurogenSegment::setZ(double _z)
{
  Z = _z;
}

void NeurogenSegment::setID(int _ID)
{
  ID = _ID;
}

void NeurogenSegment::setType(int _Type)
{
  Type = _Type;
}

void NeurogenSegment::reset()
{
  neuriteOrigin=false;
  params_p=0;
  parent_p=0;
  branch_p=0;
  resetBias();
}

void NeurogenSegment::resetBias()
{
  biasX = 0;
  biasY = 0;
  biasZ = 0;
}

void NeurogenSegment::setParentID(int _pID)
{
  Parent = _pID;
}

void NeurogenSegment::resampleGaussian(NeurogenSegment* seg_p, double sd)
{
  X = params_p->getGaussian(seg_p->getX(), sd);
  Y = params_p->getGaussian(seg_p->getY(), sd);
  Z = params_p->getGaussian(seg_p->getZ(), sd * params_p->genZ);
  if (Y>100000 || Y < -10000) {
    std::cout << "WARNING ERROR! Bad Gaussian value!" << std::endl;
  }
}

void NeurogenSegment::resampleGaussian(NeurogenSegment* seg_p)
{
  resampleGaussian(seg_p, params_p->gaussSD);
}


void NeurogenSegment::growSameDirectionAsParent()
{
  assert(parent_p->parent_p);
  double deltaX = parent_p->getX() - parent_p->parent_p->getX();
  double deltaY = parent_p->getY() - parent_p->parent_p->getY();
  double deltaZ = parent_p->getZ() - parent_p->parent_p->getZ();

  X = parent_p->getX() + deltaX;
  Y = parent_p->getY() + deltaY;
  Z = parent_p->getZ() + deltaZ;
}

void NeurogenSegment::resampleAfterForces()
{
  resampleAfterForces(params_p->gaussSD);
}

void NeurogenSegment::resampleAfterForces(double sd)
{
  X = params_p->getGaussian(parent_p->getX() + biasX, sd);
  Y = params_p->getGaussian(parent_p->getY() + biasY, sd);
  Z = params_p->getGaussian(parent_p->getZ() + biasZ, sd * params_p->genZ);
}

void NeurogenSegment::setLength(double length)
{
  double deltaX = X - parent_p->getX();
  double deltaY = Y - parent_p->getY();
  double deltaZ = Z - parent_p->getZ();

  if (fabs(deltaX)<SMALL_NUM && fabs(deltaY)<SMALL_NUM && fabs(deltaZ)<SMALL_NUM) {
    deltaX=params_p->getGaussian(0.0, SMALL_SD);
    deltaY=params_p->getGaussian(0.0, SMALL_SD);
    deltaZ=params_p->getGaussian(0.0, SMALL_SD);
    std::cerr<<"WARNING: zero length scaling. Introducing minimum SD for growth : NeurogenSegment!"<<std::endl;
  }

  //std::cout << "newX " << newX << std::endl;
  double grownDistance =  (sqrt((deltaX)*(deltaX) + (deltaY)*(deltaY) + (deltaZ)*(deltaZ)));

#ifdef DBG
  std::cout << "gD: " << grownDistance << std::endl;
  std::cout << "\n length is: " << length << std::endl;
  std::cout << deltaX << std::endl;
  std::cout << "delta Y is: " << deltaY << "\t Y=" << Y << " parent Y " <<  parent_p->getY() << std::endl;
  std::cout << deltaZ << std::endl;
  std::cout << "Parent " << parent_p->outputLine() << std::endl;
#endif
  deltaX = deltaX / grownDistance * length;
  deltaY = deltaY / grownDistance * length;
  deltaZ = deltaZ / grownDistance * length;

  X = parent_p->getX() + deltaX;
  Y = parent_p->getY() + deltaY;
  Z = parent_p->getZ() + deltaZ;
}

void NeurogenSegment::homotypicRepulsion(ShallowArray<NeurogenSegment*>& otherSegs)
{
  // K produces standard neuron at repulsion = 1.0
  double  K = 0.022;
  double deltaX = 0;
  double deltaY = 0;
  double deltaZ = 0;

  int n = otherSegs.size();
  for (int i=0; i<n; i++) {
    deltaX = X - otherSegs[i]->getX();
    deltaY = Y - otherSegs[i]->getY();
    deltaZ = Z - otherSegs[i]->getZ();
    
    double fac = K * params_p->homotypicRepulsion / pow(getDistance(otherSegs[i]), params_p->homotypicDistanceExp);

    biasX += deltaX * fac;
    biasY += deltaY * fac;
    biasZ += deltaZ * fac;
  }
}

void NeurogenSegment::tissueBoundaryRepulsion(std::map<std::string, BoundingSurfaceMesh*>& boundingSurfaceMap)
{
  BoundingSurfaceMesh* bsm = boundingSurfaceMap[params_p->boundingSurface];
  // K produces standard neuron at repulsion = 1.0
  double  K = 100000.0;
  double deltaX = 0;
  double deltaY = 0;
  double deltaZ = 0;
  for (int i=0; i<bsm->_npts; ++i) {
    deltaX = X - bsm->_hx[i];
    deltaY = Y - bsm->_hy[i];
    deltaZ = Z - bsm->_hz[i];
    
    double dist = sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ);
    double fac = K * params_p->boundaryRepulsion / pow(dist, params_p->boundaryDistanceExp);
      
    biasX += deltaX * fac;
    biasY += deltaY * fac;
    biasZ += deltaZ * fac;
  }
}

void NeurogenSegment::waypointAttraction()
{
  // K produces standard neuron at repulsion = 1.0
  double K = 0.04;
  ShallowArray<double>& waypoint = branch_p->getWaypoint1(); 
  if (waypoint.size()==3) {
    double deltaX = X - waypoint[0];
    double deltaY = Y - waypoint[1];
    double deltaZ = Z - waypoint[2];
    
    double fac = K * params_p->waypointAttraction / pow(getDistance(waypoint), params_p->waypointDistanceExp);
    
    biasX -= deltaX * fac;
    biasY -= deltaY * fac;
    biasZ -= deltaZ * fac;
  }
  else assert(waypoint.size()==0);
}

void NeurogenSegment::somaRepulsion(NeurogenSegment* soma_p)
{
  // K produces standard neuron at repulsion = 1.0
  double K = 0.04;
  double deltaX = X - soma_p->getX();
  double deltaY = Y - soma_p->getY();
  double deltaZ = Z - soma_p->getZ();

  double fac = K * params_p->somaRepulsion / pow(getDistance(soma_p), params_p->somaDistanceExp);

  biasX += deltaX * fac;
  biasY += deltaY * fac;
  biasZ += deltaZ * fac;
}

void NeurogenSegment::forwardBias()
{
  // K produces standard neuron at repulsion = 1.0
  double K = 0.3;

  double deltaX = X - parent_p->getX();
  double deltaY = Y - parent_p->getY();
  double deltaZ = Z - parent_p->getZ();

  double fac = K * params_p->forwardBias;

  biasX += deltaX * fac;
  biasY += deltaY * fac;
  biasZ += deltaZ * fac;
}

double NeurogenSegment::getSideArea()
{
  return 2*M_PI*Radius * getLength();
}

double NeurogenSegment::getVolume()
{
  return M_PI * Radius * Radius * getLength();
}

NeurogenSegment& NeurogenSegment::operator=(const NeurogenSegment &s)
{
  set(s);
  return (*this);
}

NeurogenSegment::~NeurogenSegment() {
}

