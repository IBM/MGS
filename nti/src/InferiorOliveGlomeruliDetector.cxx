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
// =================================================================

// Initiated by Heraldo Memelli 11-06-2013

#include "InferiorOliveGlomeruliDetector.h"
#include "TissueContext.h"
#include "rndm.h"

#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <cassert>
#include <float.h>
#include "ShallowArray.h"
#include "VolumeDecomposition.h"

#define MinTouchesPerGlomerulus 3
#define GlomerulusRadius 10.0
#define MinGlomeruliSpacing 10.0

InferiorOliveGlomeruliDetector::InferiorOliveGlomeruliDetector(
    TissueContext* tissueContext)
    : _tissueContext(tissueContext),
      _touchVector(0),
      _glomerulusRadiusSquared(GlomerulusRadius * GlomerulusRadius),
      _minGlomeruliSpacingSquared(MinGlomeruliSpacing * MinGlomeruliSpacing) {
  _rng.reSeed(lrandom(_rng), _tissueContext->getRank());
}

InferiorOliveGlomeruliDetector::~InferiorOliveGlomeruliDetector() {}

void InferiorOliveGlomeruliDetector::getTouchPoint(point& p, Touch& t) {
  Capsule* capsules = _tissueContext->_capsules;
  Capsule& c1 = capsules[_tissueContext->getCapsuleIndex(t.getKey1())];
  Capsule& c2 = capsules[_tissueContext->getCapsuleIndex(t.getKey2())];
  double* c1A = c1.getBeginCoordinates();
  double* c1B = c1.getEndCoordinates();
  double* c2A = c2.getBeginCoordinates();
  double* c2B = c2.getEndCoordinates();
  double pr1 = t.getProp1();
  double pr2 = t.getProp2();
  p.x = 0.5 * ((c1A[0] + pr1 * (c1B[0] - c1A[0])) +
               (c2A[0] + pr2 * (c2B[0] - c2A[0])));
  p.y = 0.5 * ((c1A[1] + pr1 * (c1B[1] - c1A[1])) +
               (c2A[1] + pr2 * (c2B[1] - c2A[1])));
  p.z = 0.5 * ((c1A[2] + pr1 * (c1B[2] - c1A[2])) +
               (c2A[2] + pr2 * (c2B[2] - c2A[2])));
}

double InferiorOliveGlomeruliDetector::getTouchDistSqrd(point_t& p, Touch& t) {
  point_t p2;
  getTouchPoint(p2, t);
  return p.DistSqrd(p2);
}

void InferiorOliveGlomeruliDetector::judgeTouches(std::list<int>& sheep,
                                                  std::list<int>& goats) {
  std::map<double, std::list<int> >::iterator miter =
      _touchMap.find((*_touchVector)[sheep.back()].getKey1());
  assert(miter != _touchMap.end());
  std::list<int>::iterator liter = miter->second.begin(),
                           lend = miter->second.end();
  for (; liter != lend; ++liter) {
    std::list<int>::iterator giter = find(goats.begin(), goats.end(), *liter);
    if (giter != goats.end()) {
      sheep.push_back(*giter);
      goats.erase(giter);
      judgeTouches(sheep, goats);
    }
  }
}

void InferiorOliveGlomeruliDetector::sparsifyTouches(std::list<int>& sheep,
                                                     std::list<int>& goats) {
  std::vector<TouchStatus> touchStatus;
  touchStatus.resize(sheep.size(), AVAILABLE);
  int nGoats = goats.size();
  std::list<int>::iterator liter1, liter2, lend = sheep.end();
  int i, j;
  for (i = 0, liter1 = sheep.begin();
       liter1 != lend && touchStatus[i] == AVAILABLE; ++i) {
    Touch& t1 = (*_touchVector)[*liter1];
    for (j = i + 1, liter2 = ++liter1;
         liter2 != lend && touchStatus[j] == AVAILABLE; ++j, ++liter2) {
      Touch& t2 = (*_touchVector)[*liter2];
      if (isReciprocal(t1, t2)) continue;
      unsigned int nid1A = _segmentDescriptor.getNeuronIndex(t1.getKey1());
      unsigned int nid1B = _segmentDescriptor.getNeuronIndex(t1.getKey2());
      unsigned int nid2A = _segmentDescriptor.getNeuronIndex(t2.getKey1());
      unsigned int nid2B = _segmentDescriptor.getNeuronIndex(t2.getKey2());
      if (((nid1A == nid2A) && (nid1B == nid2B)) ||
          ((nid1A == nid2B) && (nid1B == nid2A))) {
        goats.push_back(*liter2);
        touchStatus[j] = EXCLUDED;
      }
    }
  }
  if (goats.size() > nGoats) {
    std::list<int> survivingSheep;
    for (i = 0, liter1 = sheep.begin(); liter1 != lend; ++i, ++liter1) {
      if (touchStatus[i] == AVAILABLE) survivingSheep.push_back(*liter1);
    }
    sheep = survivingSheep;
  }
}

bool InferiorOliveGlomeruliDetector::isReciprocal(Touch& t1, Touch& t2) {
  return ((t1.getKey1() == t2.getKey2()) && (t1.getKey2() == t2.getKey1()));
}

bool InferiorOliveGlomeruliDetector::checkGlomeruli(double* constraints) {
  bool moreToDo = false;
  point_t p;
  p.x = constraints[0];
  p.y = constraints[1];
  p.z = constraints[2];
  TouchVector& touches = *_touchVector;
  int nrTouches = touches.getCount();
  std::list<int> nonCenter;
  for (int i = 0; i < nrTouches; ++i) {
    double d = getTouchDistSqrd(p, touches[i]);
    if (d < _minGlomeruliSpacingSquared) {
      nonCenter.push_back(i);
      _touchStatus[i] = NONCENTER;
      if (d < _glomerulusRadiusSquared) _touchStatus[i] = EXCLUDED;
    }
  }
  std::list<int>::iterator iter = nonCenter.begin(), end = nonCenter.end();
  for (; iter != end; ++iter) {
    std::map<int, glomerulus_t>::iterator giter = _glomeruli.find(*iter);
    if (giter != _glomeruli.end()) {
      assert(giter->second.score != constraints[3]);
      if (giter->second.score < constraints[3]) {
        moreToDo = true;
        std::list<int>::iterator titer, tend = giter->second.included.end();
        for (titer = giter->second.included.begin(); titer != tend; ++titer) {
          if (_touchStatus[*titer] == INCLUDED)
            _touchStatus[*titer] = AVAILABLE;
        }
        _glomeruli.erase(giter);
      }
    }
  }
  return moreToDo;
}

void InferiorOliveGlomeruliDetector::findGlomeruli(TouchVector* touchVector) {
  int rank = _tissueContext->getRank();
  int mpiSize = _tissueContext->getMpiSize();

  if (_tissueContext->getRank() == 0) printf("Detecting Glomeruli...\n\n");
  ;
  _touchVector = touchVector;
  TouchVector& touches = *_touchVector;
  int nrTouches = touches.getCount();

  std::vector<point_t> touchPoints;
  std::map<double, int> randomizer;

  touchPoints.resize(nrTouches);
  for (int i = 0; i < nrTouches; ++i) {
    getTouchPoint(touchPoints[i], touches[i]);
    _touchMap[touches[i].getKey1()].push_back(i);
    randomizer[drandom(_rng)] = i;
  }
  assert(randomizer.size() == nrTouches);

  std::vector<int> touchOrder;
  std::map<double, int>::iterator miter = randomizer.begin(),
                                  mend = randomizer.end();
  for (; miter != mend; ++miter) touchOrder.push_back(miter->second);

  _touchStatus.resize(nrTouches, AVAILABLE);

  bool globalDone = false, localDone = false;
  MPI_Request request;
  double recvbuf[4];
  double sendbuf[4];
  MPI_Irecv(recvbuf, 4, MPI_DOUBLE, MPI_ANY_SOURCE,
	    MPI_ANY_TAG, MPI_COMM_WORLD, &request);
  int sentReceived[2] = {0, 0};
  Sphere sphere;
  sphere._radius = GlomerulusRadius + MinGlomeruliSpacing;
  sphere._key = sphere._dist2Soma = 0;

  while (!globalDone) {
    if (!localDone) {
      localDone = true;
      for (int i = 0; i < nrTouches; ++i) {
        int ctrIdx = touchOrder[i];
        while (_touchStatus[ctrIdx] != AVAILABLE && ++i < nrTouches)
          ctrIdx = touchOrder[i];
        if (i >= nrTouches) break;

        Touch& ctrTouch = touches[ctrIdx];
        point_t ctr;
        getTouchPoint(ctr, ctrTouch);

        glomerulus_t glom(ctr);
        std::list<int> candidates, nonCenter;
        glom.included.push_back(ctrIdx);
        for (int j = 0; j < nrTouches; ++j) {
          int candidateIdx = touchOrder[j];
          if (candidateIdx == ctrIdx) continue;
          if (_touchStatus[candidateIdx] == AVAILABLE ||
              _touchStatus[candidateIdx] == NONCENTER) {
            double d = getTouchDistSqrd(ctr, touches[candidateIdx]);
            if (d < _minGlomeruliSpacingSquared) {
              if (d < _glomerulusRadiusSquared)
                candidates.push_back(candidateIdx);
              else
                nonCenter.push_back(candidateIdx);
            }
          }
        }
        if (candidates.size() >= MinTouchesPerGlomerulus - 1) {
          judgeTouches(glom.included, candidates);
          sparsifyTouches(glom.included, candidates);
        }
        if (glom.included.size() >= MinTouchesPerGlomerulus) {
          _touchStatus[ctrIdx] = CENTER;
          std::list<int>::iterator liter = glom.included.begin(),
                                   lend = glom.included.end();
          assert(ctrIdx == *liter);
          for (++liter; liter != lend; ++liter) _touchStatus[*liter] = INCLUDED;
          lend = candidates.end();
          for (liter = candidates.begin(); liter != lend; ++liter)
            _touchStatus[*liter] = EXCLUDED;
          lend = nonCenter.end();
          for (liter = nonCenter.begin(); liter != lend; ++liter)
            _touchStatus[*liter] = NONCENTER;
          glom.score = drandom(_rng);
          assert(_glomeruli.find(ctrIdx) == _glomeruli.end());
          _glomeruli[ctrIdx] = glom;

          ShallowArray<int, MAXRETURNRANKS, 100> ranks;
          std::copy(&ctr.x, &ctr.x + 3, sphere._coords);
          // memcpy(sphere._coords, &ctr.x, sizeof(double)*3);
          _tissueContext->_decomposition->getRanks(&sphere, 0.0, ranks);
          if (ranks.size() > 1) {
            std::copy(&ctr.x, &ctr.x + 3, sendbuf);
            // memcpy(sendbuf, &ctr.x, sizeof(double) * 3);
            sendbuf[3] = glom.score;
            ShallowArray<int>::iterator riter = ranks.begin(),
                                        rend = ranks.end();
            for (; riter != rend; ++riter) {
              if (*riter == rank) continue;
              MPI_Isend(sendbuf, 4, MPI_DOUBLE, *riter, 0, MPI_COMM_WORLD, &request);
              ++sentReceived[0];
            }
          }
        }
        MPI_Status status;
        bool moreToDo = false;
	int flag;
	MPI_Test(&request, &flag, &status);
        while (flag) {
          ++sentReceived[1];
          moreToDo = (moreToDo || checkGlomeruli(recvbuf));
          MPI_Irecv(recvbuf, 4, MPI_DOUBLE,
		    MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
	  MPI_Test(&request, &flag, &status);
        }
        localDone = (localDone && !moreToDo);
      }
    }
    int totalSentReceived[2];
    MPI_Allreduce((void*)sentReceived, (void*)totalSentReceived, 2,
		  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    localDone = globalDone = (totalSentReceived[0] == totalSentReceived[1]);
  }
  MPI_Cancel(&request);
  int totalCount;
  ;
  int nGlomeruli = _glomeruli.size();
  MPI_Allreduce((void*)&nGlomeruli, (void*)&totalCount, 1, MPI_INT,
		MPI_SUM, MPI_COMM_WORLD);
  if (_tissueContext->getRank() == 0)
    printf("Total Glomeruli = %d\n\n", totalCount);
  TouchVector newTouchVector;
  for (int i = 0; i < nrTouches; ++i)
    if (_touchStatus[i] == INCLUDED || _touchStatus[i] == CENTER)
      newTouchVector.push_back(touches[i], 0);
  int nrGlomeruliTouches = newTouchVector.getCount();
  totalCount = 0;
  MPI_Allreduce((void*)&nrGlomeruliTouches, (void*)&totalCount, 1,
		MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (_tissueContext->getRank() == 0)
    printf("Total Glomeruli Touches = %d\n\n", totalCount);
  touches.clear();
  for (int i = 0; i < nrGlomeruliTouches; ++i)
    touches.push_back(newTouchVector[i], 0);
}

double InferiorOliveGlomeruliDetector::getGlomeruliSpacing() {
  return MinGlomeruliSpacing;
}
