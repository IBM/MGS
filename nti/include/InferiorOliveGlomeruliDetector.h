// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

// by Heraldo Memelli 11-06-2013

#ifndef INFERIOROLIVEGLOMERULIDETECTOR_H
#define INFERIOROLIVEGLOMERULIDETECTOR_H

#include <mpi.h>
#include "GlomeruliDetector.h"
#include "TouchVector.h"
#include "SegmentDescriptor.h"
#include "rndm.h"
#include <map>

class TissueContext;

class InferiorOliveGlomeruliDetector : public GlomeruliDetector
{
  private:
    typedef class point
    {
      public:
        point() : x(0), y(0), z(0) {}
        point(point const & p) : x(p.x), y(p.y), z(p.z) {}
	double DistSqrd(point& p2)
	{
	  return (p2.x-x)*(p2.x-x)+(p2.y-y)*(p2.y-y)+(p2.z-z)*(p2.z-z);
	}
	double x, y, z;
    } point_t;

    typedef enum{AVAILABLE=0, EXCLUDED, NONCENTER, INCLUDED, CENTER} TouchStatus;

    void getTouchPoint(point& p, Touch& t);
    double getTouchDistSqrd(point_t& p, Touch& t);
    void judgeTouches(std::list<int>& sheep, std::list<int>& goats);
    void sparsifyTouches(std::list<int>& sheep, std::list<int>& goats);
    bool isReciprocal(Touch& t1, Touch& t2);
    bool checkGlomeruli(double* constraints);
    TissueContext* _tissueContext;
    TouchVector* _touchVector;
    std::map<double, std::list<int> > _touchMap;
    std::vector<TouchStatus> _touchStatus;
    double _glomerulusRadiusSquared, _minGlomeruliSpacingSquared;
    SegmentDescriptor _segmentDescriptor;
    RNG _rng;

  public:	
    typedef class glomerulus
    {
      public:
        glomerulus() : score(0) {}
        glomerulus(point_t p) : center(p), score(0) {}
        glomerulus(glomerulus const & g) : center(g.center), included(g.included), score(g.score) {}
	point_t center;
	std::list<int> included;
	double score;
    } glomerulus_t;

    InferiorOliveGlomeruliDetector(TissueContext*);
    virtual ~InferiorOliveGlomeruliDetector();
    void findGlomeruli(TouchVector*);
    double getGlomeruliSpacing();
    std::map<int, glomerulus_t> _glomeruli;
};

#endif
