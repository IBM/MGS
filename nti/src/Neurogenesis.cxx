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
 * Neurogenesis.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: heraldo
 */

#include <mpi.h>

#include "Neurogenesis.h"
#include "NeurogenBranch.h"
#include "NeurogenSegment.h"
#include "BoundingCuboid.h"
#include "BoundingSurfaceMesh.h"
#include "NeurogenParams.h"
#include "BuffFactor.h"
#include "WaypointGenerator.h"
#include "OlivoCerebellarWaypointGenerator.h"
#include "TestWaypointGenerator.h"
#ifndef DISABLE_PTHREADS
#include "For.h"
#endif
#include <fstream>
#include <float.h>

#define APPLY_FORCES_TO_BRANCHES
#define CONST_BR_PROB

//#define POLARIZED_NEURON

Neurogenesis::Neurogenesis(
    int rank, int size, int nThreads, std::string statsFileName,
    std::string parsFileName, bool stdout, bool fout, int branchType,
    std::map<std::string, BoundingSurfaceMesh*>& boundingSurfaceMap)
    : NeuronBoundary(0),
      totalBifurcations(0),
      _neuronBegin(0),
      _neurogenParams(0),
      _neurogenFileNames(0),
      _statsFileName(statsFileName),
      _parsFileName(parsFileName),
      _stdout(stdout),
      _fout(fout),
      _somaGenerated(0),
      _branchType(branchType),
      _boundingSurfaceMap(boundingSurfaceMap),
      _rank(rank),
      _size(size),
      _nThreads(nThreads),
      _threadData(0) {
  WaypointGeneratorMap["OlivoCerebellar"] =
      new OlivoCerebellarWaypointGenerator();
  WaypointGeneratorMap["Test"] = new TestWaypointGenerator();
  NeuronBoundary = new BoundingCuboid();
  _threadData = new ThreadData[_nThreads];
}

void Neurogenesis::run(int neuronBegin, int nNeurons, NeurogenParams** params,
                       std::vector<std::string>& fileNames,
                       bool* somaGenerated) {
  std::map<std::string, WaypointGenerator*>::iterator miter,
      mend = WaypointGeneratorMap.end();
  for (miter = WaypointGeneratorMap.begin(); miter != mend; ++miter) {
    assert(miter->second);
    miter->second->readWaypoints(fileNames, nNeurons);
  }

  if (_rank == 0 && _fout) {
    std::ofstream fout;
    NeurogenParams p(_rank);
    fout.open(_parsFileName.c_str(), std::ios::out);
    p.printParams(fout, true, true, "", "\t", "\t");
    fout.close();
    fout.open(_statsFileName.c_str(), std::ios::out);
    printStats(0, 0, &p, fout, 0, true, true, "\t", "\t");
    fout.close();
  }

  if (nNeurons > 0) {
    _neurogenParams = params;
    _neuronBegin = neuronBegin;
    _neurogenFileNames = fileNames;
    _somaGenerated = somaGenerated;

#ifndef DISABLE_PTHREADS
    For<Neurogenesis, ThreadData>::execute(0, 1, nNeurons, this, _threadData,
                                           _nThreads);
#else
    for (int nid = 0; nid < nNeurons; ++nid) {
      doWork(0, nid, &_threadData[nid], 0);
    }
#endif
  }
  if (_fout) {
    int nextToWrite = 0, written = 0;
    while (nextToWrite < _size) {
      if (nextToWrite == _rank) {
        if (nNeurons > 0) {
          std::ofstream fout;
          fout.open(_parsFileName.c_str(), std::ios::app);
          fout << _opars.str();
          fout.close();
          fout.open(_statsFileName.c_str(), std::ios::app);
          fout << _ostats.str();
          fout.close();
        }
        written = 1;
      }
      MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    _opars.str("");
    _ostats.str("");
  }
}

void Neurogenesis::doWork(int threadID, int nid, ThreadData* data,
                          Mutex* mutex) {
  if (_neurogenParams[nid]) {
    data[threadID].reset(_neurogenParams[nid]->maxSegments, nid, mutex);
    int& nrStemSegments = data[threadID].nrStemSegments;
    generateArbors(threadID, nid, nrStemSegments);
    _somaGenerated[nid] = true;
    if (_fout || _stdout) {
#ifndef DISABLE_PTHREADS
      mutex->lock();
#endif
      if (_fout) {
        _neurogenParams[nid]->printParams(_opars, false, true, "", "\t", "\t");
        printStats(threadID, _neuronBegin + nid, _neurogenParams[nid], _ostats,
                   nrStemSegments, false, true, "\t", "\t");
      }
      if (_stdout) {
        std::ostringstream os;
        os << "PARAMS-" << _parsFileName << " : " << _neuronBegin + nid
           << std::endl;
        _neurogenParams[nid]->printParams(os, true, true, "", " : ", "\n");
        printStats(threadID, _neuronBegin + nid, _neurogenParams[nid], os,
                   nrStemSegments, true, true, " : ", "\n");
        std::cerr << os.str() << std::endl;
      }
#ifndef DISABLE_PTHREADS
      mutex->unlock();
#endif
    }
  }
}

void Neurogenesis::generateArbors(int threadID, int nid, int& nrStemSegments) {
  NeurogenParams* params_p = _neurogenParams[nid];
#ifndef _SILENT_
  std::cout << "Generating Neuron " << nid << std::endl;
#endif
  // createTestRotation();
  int count = 1;
  if (_somaGenerated[nid])
    readSoma(threadID, nid, count, params_p);
  else
    generateSoma(threadID, params_p, count);

  // Start actual generation of neuritic branches

  generateStems(threadID, params_p, nid, nrStemSegments);
  // Generate everything
  extendNeurites(threadID, params_p);
  setSegmentIDs(threadID, params_p, count);
  writeSWC(threadID, nid, _neurogenFileNames[nid].c_str());
}

void Neurogenesis::readSoma(int threadID, int nid, int& count,
                            NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  NeurogenBranch* soma = data.allocBranch();
  NeurogenSegment* first = data.allocSegment(params_p);
  first->setBranch(soma);
  first->setStartingCoords();

  int id, type, parent;
  double x, y, z, r;

  FILE* inputDataFile;
  // char* filename = _neurogenFileNames[nid];
  std::string filename = _neurogenFileNames[nid];
  if ((inputDataFile = fopen(filename.c_str(), "rt")) == 0) {
    printf("Could not find the input file %s...\n", filename.c_str());
    MPI_Finalize();
    exit(0);
  }
  while (fscanf(inputDataFile, "%d %d %lf %lf %lf %lf %d", &id, &type, &x, &y,
                &z, &r, &parent) != EOF) {
    if (type == 1) {
      if (first) {
        first->setRadius(r);
        first->setID(count);
        soma->addSegment(first);
        first = 0;
      } else {
        NeurogenSegment* seg_p =
            data.allocSegment(count, 1, x, y, z, r, 1, params_p);
        seg_p->setParent(&data.Segments[0]);
        soma->addSegment(seg_p);
      }
    }
    ++count;
  }
  fclose(inputDataFile);
}

void Neurogenesis::generateSoma(int threadID, NeurogenParams* params_p,
                                int& count) {
  ThreadData& data = _threadData[threadID];
  double somaRadius = sqrt(params_p->somaSurface / (4 * M_PI));

  NeurogenBranch* soma = data.allocBranch();
  NeurogenSegment* first = data.allocSegment(params_p);
  soma->setSoma(first);
  first->setBranch(soma);
  first->setStartingCoords();
  first->setRadius(somaRadius);
  first->setID(count);
  ++count;
  soma->addSegment(first);
  for (int i = 0; i < params_p->somaSegments; i++) {
    NeurogenSegment* seg_p = data.allocSegment(
        i + 2, 1, params_p->startX, params_p->startY + i * somaRadius,
        params_p->startZ, somaRadius, 1, params_p);
    seg_p->setParent(&data.Segments[0]);
    seg_p->setID(count);
    ++count;
    soma->addSegment(seg_p);
  }
}

void Neurogenesis::generateStems(int threadID, NeurogenParams* params_p,
                                 int nid, int& nrStemSegments) {
  ThreadData& data = _threadData[threadID];
  NeurogenSegment* newSeg = 0;
  NeurogenBranch* newBranch = 0;
  NeurogenSegment* soma = &data.Segments[0];

  // Generate Arbors
  int attempts = nrStemSegments = 0;
  int nrStems = 0;
  if (_boundingSurfaceMap[params_p->boundingSurface]
          ->isOutsideVolume(&data.Segments[0])) {
    std::cerr << "Neuron " << data.nid << ", (" << data.Segments[0].getX()
              << ", " << data.Segments[0].getY() << ", "
              << data.Segments[0].getZ() << "), is outside surface_mesh.vtk!"
              << std::endl;
    exit(0);
  }

  while (nrStems < params_p->nrStems) {
    // std::cout << "Generating Dendrites" << i << std::endl;
    newBranch = data.allocBranch();
    newBranch->setSoma(soma);

#ifdef POLARIZED_NEURON
    if (nrStems == 0)
      // Create Dendrite in Y direction
      newSeg = data.allocSegment(_branchType, params_p->startX,
                                 params_p->startY + 1.0, params_p->startZ,
                                 params_p->startRadius, params_p);
    else
#endif
      // Create Neurite in Random direction
      newSeg = data.allocSegment(
          _branchType, params_p->getGaussian(data.Segments[0].getX(), 1.0),
          params_p->getGaussian(data.Segments[0].getY(), 1.0),
          params_p->getGaussian(data.Segments[0].getZ(), 1.0 * params_p->genZ),
          params_p->startRadius, params_p);

    newSeg->setParent(&data.Segments[0]);
    newSeg->setLength(sqrt(params_p->somaSurface / (4 * M_PI)));
    int resampleCount = 0;
    // simple rejection sampling to make sure that the dendrite stems are not
    // too close to each other
    int resamples = 0;

    while (resamples < 360 &&
           (withinMinimumStemAngleOfAnotherSeg(
                threadID, newSeg, double(360 - resamples) / 360.0, params_p) ||
            touchesAnotherSeg(threadID, newSeg, params_p))) {
      newSeg->resampleGaussian(&data.Segments[0], 1.0);
      ++resamples;
    }
    if (checkBorders(newSeg)) {
      data.deleteSegment();
      data.deleteBranch();
      ++attempts;
    } else {
      newSeg->setLength(sqrt(params_p->somaSurface / (4 * M_PI)));
      ++nrStems;
      ++nrStemSegments;
      newSeg->setBranch(newBranch);
      newBranch->addSegment(newSeg);
    }
    if (attempts >= 1000) {
      std::cerr << "ERROR: Only " << nrStems << " type " << _branchType
                << " stems created!" << std::endl;
      exit(0);
    }

    WaypointGenerator* waypointGenerator =
        WaypointGeneratorMap[params_p->waypointGenerator];
    if (waypointGenerator) {
      ShallowArray<ShallowArray<double> > waypointCoords;
      waypointGenerator->next(waypointCoords, nid);
      assert(nrStems <= waypointCoords.size());
      int nrTerminals = waypointCoords.size();
      int nrGrowingStems = 0;
      for (int i = 0; i < waypointCoords.size(); ++i) {
        if (waypointCoords[i].size() > 0) {
          ++nrGrowingStems;
          if (i > 0) assert(waypointCoords[i - 1].size() > 0);
        }
      }
      assert(nrGrowingStems == nrStems);
      ShallowArray<NeurogenSegment*> origins;
      origins.increaseSizeTo(nrTerminals);
      for (int i = 0; i < nrTerminals; ++i) {
        if (i < nrStems)
          origins[i] = &data.Segments[data.nrSegments - nrStems + i];
        else
          origins[i] = 0;
      }
      bool done = false;
      while (!done) {
        int nrDoneGrowing = 0;
        std::map<double, int> branchCandidates;
        for (int i = 0; i < nrTerminals; ++i) {
          if (waypointCoords[i].size() > 0) {
            if (origins[i] != 0) {
              origins[i]->getBranch()->addWaypoint(waypointCoords[i]);
            } else {
              for (int j = 0; j < i; ++j) {
                if (waypointCoords[j].size() > 0) {
                  branchCandidates[origins[j]->getDistance(waypointCoords[i])] =
                      j;
                  origins[j]->getBranch()->addWaypoint(waypointCoords[i]);
                }
              }
            }
          } else if (origins[i] != 0)
            ++nrDoneGrowing;
        }

        done = (nrDoneGrowing == nrTerminals);
        if (branchCandidates.size() > 0) {
          int branch1, branch2 = nrGrowingStems;
          std::map<double, int>::iterator branchCandidate =
                                              branchCandidates.begin(),
                                          candidatesEnd =
                                              branchCandidates.end();
          bool success = false;
          while (branchCandidate != candidatesEnd && !success) {
            branch1 = branchCandidate->second;
            success = branchWithAngle(threadID, origins[branch1], params_p);
            ++branchCandidate;
          }
          if (!success) {
            std::cerr << "Neurogenesis : Branching failed on rank "
                      << params_p->_rank
                      << " during waypoint guided stem extension of neuron "
                      << nid << "!" << std::endl;
            MPI_Finalize();
            exit(0);
          }
          nrStemSegments += 2;
          nrGrowingStems++;
          origins[branch1] = &data.Segments[data.nrSegments - 2];
          origins[branch2] = &data.Segments[data.nrSegments - 1];
        }

        // number of nrGrowingStems now equals waypointCoords.size()
        for (int i = 0; i < nrGrowingStems; ++i) {
          if (waypointCoords[i].size() > 0) {
            double dist = DBL_MAX;
            NeurogenSegment* tip = origins[i];
            while (dist > params_p->waypointExtent &&
                   growSegment(threadID, tip, params_p)) {
              nrStemSegments++;
              tip = &data.Segments[data.nrSegments - 1];
              dist = tip->getDistance(tip->getBranch()->getWaypoint1());
            }
            origins[i] = tip;
          }
        }
        waypointGenerator->next(waypointCoords, nid);
        for (int i = 0; i < nrGrowingStems; ++i)
          origins[i]->getBranch()->clearWaypoints();
      }
      for (int i = 0; i < nrTerminals; ++i) origins[i]->setNeuriteOrigin(true);
    } else {
      data.Segments[data.nrSegments - 1].setNeuriteOrigin(true);
    }
  }
}

void Neurogenesis::extendNeurites(int threadID, NeurogenParams* params) {
  ThreadData& data = _threadData[threadID];
  NeurogenParams params_p = *params;

  if (params->terminalField != "NULL")
    params_p.load(params->terminalField, params->_rank);
  int b = 0;
  for (int i = 1; i < data.nrSegments; ++i) {
    if (data.Segments[i].isNeuriteOrigin()) {
      data.Segments[i].getBranch()->clearWaypoints();
      if (params->terminalField != "NULL") {
        ShallowArray<double> rs;
        rs.push_back(data.Segments[i].getX());
        rs.push_back(data.Segments[i].getY());
        rs.push_back(data.Segments[i].getZ());
        data.Segments[i].getBranch()->setSoma(&data.Segments[i]);
        params_p.startX = data.Segments[i].getX();
        params_p.startY = data.Segments[i].getY();
        params_p.startZ = data.Segments[i].getZ();
      }
      growSegment(threadID, &data.Segments[i], &params_p);
    }
  }

  // Go Round-Robin through all the segments and grow the active ones
  for (unsigned int j = 1 + data.nrStemSegments; j < data.nrSegments; j++) {
#ifdef CONST_BR_PROB
    double branchProb =
        1.0 - pow(1.0 - params_p.branchProb, data.Segments[j].getLength());
#else
    double branchProb = params_p.branchProb;
#endif

    if (params->terminalField != "NULL") {
      NeurogenSegment* soma = data.Segments[j].getBranch()->getSoma();
      params_p.startX = soma->getX();
      params_p.startY = soma->getY();
      params_p.startZ = soma->getZ();
    }

    if (drandom(0.0, 1.0, params_p._rng) < branchProb &&
        totalBifurcations <= params_p.maxBifurcations) {
      branchWithAngle(threadID, &data.Segments[j], &params_p);
    } else
      growSegment(threadID, &data.Segments[j], &params_p);

    if (data.nrSegments >= params_p.maxSegments) {
#ifdef DBG
      std::cout << "Max Number of Segments reached..." << std::endl;
#endif
      break;
    }

    if (getTotalFiberLength(threadID, &params_p) > params_p.maxFiberLength) {
#ifdef DBG
      std::cout << "Max Fiber Length reached..." << std::endl;
      std::cout << "Total segments: " << data.nrSegments << std::endl;
#endif
      break;
    }
  }
}

bool Neurogenesis::branchWithAngle(int threadID, NeurogenSegment* seg_p,
                                   NeurogenParams* params_p) {
  bool rval = false;
  ThreadData& data = _threadData[threadID];
  NeurogenBranch* branch1 = data.allocBranch();
  NeurogenBranch* branch2 = data.allocBranch();
  branch1->setWaypoint1(seg_p->getBranch()->getWaypoint1());
  branch2->setWaypoint1(seg_p->getBranch()->getWaypoint2());
  NeurogenSegment* soma = seg_p->getBranch()->getSoma();
  branch1->setSoma(soma);
  branch2->setSoma(soma);

  double newRadius1 = seg_p->getRadius() * params_p->RallsRatio * 0.5;
  double fac =
      pow(params_p->minRadius / newRadius1, params_p->branchDiameterAsymmetry);

  if (fac < 1.0) {
    double fac2 =
        params_p->radiusRate * (1.0 - fac) + 0.5 * params_p->RallsRatio * fac;
    newRadius1 = seg_p->getRadius() * fac2;
  }

  double newRadius2 = seg_p->getRadius() * params_p->RallsRatio - newRadius1;

  if (drandom(0.0, 1.0, params_p->_rng) < 0.5) {
    double tmp = newRadius1;
    newRadius1 = newRadius2;
    newRadius2 = tmp;
  }

  NeurogenSegment* child1 =
      data.allocSegment(seg_p->getType(), seg_p->getX(), seg_p->getY(),
                        seg_p->getZ(), newRadius1, params_p);
  child1->setParent(seg_p);
  child1->growSameDirectionAsParent();

  NeurogenSegment* child2 =
      data.allocSegment(seg_p->getType(), seg_p->getX(), seg_p->getY(),
                        seg_p->getZ(), newRadius2, params_p);
  child2->setParent(seg_p);
  child2->growSameDirectionAsParent();

  child1->setBranch(branch1);
  branch1->addSegment(child1);
  child2->setBranch(branch2);
  branch2->addSegment(child2);

  double lFront1 = params_p->umnPerFront / pow(2.0 * M_PI * child1->getRadius(),
                                               params_p->nexpPerFront);
  ;
  child1->setLength(lFront1);
  double lFront2 = params_p->umnPerFront / pow(2.0 * M_PI * child2->getRadius(),
                                               params_p->nexpPerFront);
  ;
  child2->setLength(lFront2);

  if (params_p->genZ > 0.5)
    seg_p->rotateDaughters(child1, child2);
  else
    seg_p->rotateDaughters2D(child1, child2);

#ifdef APPLY_FORCES_TO_BRANCHES
  ShallowArray<NeurogenSegment*> segVec1;
  segVec1.push_back(child2);
  applyForces(threadID, child1, segVec1, params_p);

  ShallowArray<NeurogenSegment*> segVec2;
  segVec2.push_back(child1);
  applyForces(threadID, child2, segVec2, params_p);

  resampleAfterForces(child1);
  resampleAfterForces(child2);
#endif
  if (checkTouchAndRelax(threadID, child1, params_p)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Daughter Branch touching other segments! "
                 "Terminating Branch!" << std::endl;
#endif
    branch1 = 0;
  }
  if (checkTouchAndRelax(threadID, child2, params_p)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Daughter Branch touching other segments! "
                 "Terminating Branch!" << std::endl;
#endif
    branch2 = 0;
  }

  if (branch1 && params_p->genZ == 0.0 && checkIntersect2D(threadID, child1)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Daughter Branch touching other segments! "
                 "Terminating Branch!" << std::endl;
#endif
    branch1 = 0;
  }
  if (branch2 && params_p->genZ == 0.0 && checkIntersect2D(threadID, child2)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Daughter Branch touching other segments! "
                 "Terminating Branch!" << std::endl;
#endif
    branch2 = 0;
  }

  if (branch1 && checkBorders(child1)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Daughter Branch Reached the border of Volume!! " << std::endl;
#endif
    branch1 = 0;
  }
  if (branch2 && checkBorders(child2)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Daughter Branch Reached the border of Volume!! " << std::endl;
#endif
    branch2 = 0;
  }

  if (branch1 && checkRadius(child1, params_p)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Daughter Radius too small!  Stopping Branch! " << std::endl;
#endif
    branch1 = 0;
  }
  if (branch2 && checkRadius(child2, params_p)) {
#ifdef SHOWTERMINATIONS
    std::cout << "Daughter Radius too small!  Stopping Branch! " << std::endl;
#endif
    branch2 = 0;
  }

  if (branch2 == 0 && branch1 == 0) {
    data.deleteSegment();  // delete branch2
    data.deleteBranch();
    data.deleteSegment();  // delete branch1
    data.deleteBranch();
  } else if (branch2 == 0) {
    data.deleteSegment();  // delete branch2
    data.deleteBranch();
  } else if (branch1 == 0) {
    NeurogenSegment s = data.Segments[data.nrSegments - 1];  // store branch2

    data.deleteSegment();  // delete branch2
    data.deleteBranch();
    data.deleteSegment();  // delete branch1
    data.deleteBranch();

    child1 = data.allocSegment(s);  // reallocate branch2
    branch1 = data.allocBranch();
    child1->setBranch(branch1);
    branch1->addSegment(child1);
  } else {
    totalBifurcations++;
    rval = true;
  }
  return rval;
}

bool Neurogenesis::growSegment(int threadID, NeurogenSegment* seg_p,
                               NeurogenParams* params_p) {
  bool rval = false;
  ThreadData& data = _threadData[threadID];
  NeurogenSegment* newSeg =
      data.allocSegment(seg_p->getType(), seg_p->getX(), seg_p->getY(),
                        seg_p->getZ(), seg_p->getRadius(), params_p);
  double newRadius = seg_p->getRadius() * params_p->radiusRate;
  double lFront = params_p->umnPerFront /
                  pow(2.0 * M_PI * newRadius, params_p->nexpPerFront);

  newSeg->setParent(seg_p);
  newSeg->setRadius(newRadius);
  newSeg->growSameDirectionAsParent();
  newSeg->setBranch(seg_p->getBranch());
  newSeg->setLength(lFront);
  ShallowArray<NeurogenSegment*> segVec;
  applyForces(threadID, newSeg, segVec, params_p);
  resampleAfterForces(newSeg);

  int resampleCount = 0;
  if (checkTouchAndRelax(threadID, newSeg, params_p)) {
    data.deleteSegment();
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Growing Segment touching other segments! "
                 "Terminating Branch!" << std::endl;
#endif
  } else if (checkBorders(newSeg)) {
    data.deleteSegment();
#ifdef SHOWTERMINATIONS
    std::cout << "Reached the border of Volume!! " << std::endl;
#endif
  } else if (checkRadius(newSeg, params_p)) {
    data.deleteSegment();
#ifdef SHOWTERMINATIONS
    std::cout << "Radius too small!  Terminating Branch! " << std::endl;
#endif
  } else if (params_p->genZ == 0.0 && checkIntersect2D(threadID, newSeg)) {
    data.deleteSegment();
#ifdef SHOWTERMINATIONS
    std::cout << "Warning! Growing Segment intersects with another segment!  "
                 "Terminating Branch! " << std::endl;
#endif
  } else {
    rval = true;
    seg_p->getBranch()->addSegment(newSeg);
  }

#ifdef DBG
  //	std::cout << "Updated Grown Distance: " << grownDistance << " and
  //lFront: " << lFront << std::endl;
  std::cout << "Parent: " << seg_p->outputLine() << std::endl;
  std::cout << newSeg->outputLine() << std::endl;
#endif
  return rval;
}

bool Neurogenesis::touchesAnotherSeg(int threadID, NeurogenSegment* seg_p,
                                     NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  bool rval = false;
  for (unsigned int i = 1 + params_p->somaSegments;
       i < data.nrSegments && !rval; i++) {
    if (seg_p == &data.Segments[i]) continue;
    if (seg_p->touches(&data.Segments[i], 0.0)) {
      rval = true;
    }
  }
  return rval;
}

bool Neurogenesis::touchesAnotherSegWithExceptions(int threadID,
                                                   NeurogenSegment* seg_p,
                                                   NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  bool rval = false;
  for (unsigned int i = 1 + params_p->somaSegments;
       i < data.nrSegments && !rval; i++) {
    if (seg_p == &data.Segments[i]) continue;
    // std::cout << "Checking for touches " << std::endl;
    if (seg_p->getParentSeg()->equals(&data.Segments[i])) {
    }
    // std::cout << "Skipping own parent" << std::endl;
    else if (seg_p->getParentSeg()->getParentSeg() &&
             seg_p->getParentSeg()->getParentSeg()->equals(&data.Segments[i])) {
    }
    // std::cout << "Skipping grand parent" << std::endl;
    else if (seg_p->touches(&data.Segments[i], params_p->intolerance)) {
#ifdef DBG
      std::cout << "Segment: " << std::endl << seg_p->outputLine()
                << " touches " << data.Segments[i].outputLine() << std::endl;
#endif
      rval = true;
    }
  }
  return rval;
}

bool Neurogenesis::withinMinimumStemAngleOfAnotherSeg(
    int threadID, NeurogenSegment* seg_p, double factor,
    NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  bool rval = false;
  for (unsigned int i = 1 + params_p->somaSegments;
       i < data.nrSegments && !rval; i++) {
    if (seg_p == &data.Segments[i]) continue;
    if (seg_p->getAngle(&data.Segments[i]) <
        M_PI / 180.0 * params_p->minInitialStemAngle * factor) {
      rval = true;
    }
  }
  return rval;
}

void Neurogenesis::segmentsNear(int threadID,
                                ShallowArray<NeurogenSegment*>& SegTouchVec,
                                NeurogenSegment* seg_p,
                                NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  for (unsigned int i = 1 + data.nrStemSegments; i < data.nrSegments; i++) {
    if (seg_p == &data.Segments[i]) continue;
#ifdef DBG
    std::cout << "Checking for touches " << std::endl;
    if (seg_p->getParentSeg()->equals(&data.Segments[i]))
      std::cout << "Skipping own parent" << std::endl;
#endif
    if (!seg_p->getParentSeg()->equals(&data.Segments[i]) &&
        !seg_p->getParentSeg()->getParentSeg()->equals(&data.Segments[i])) {
#ifdef DBG
      std::cout << "NEWSEG: " << seg_p->outputLine()
                << " is within the homotypic distance "
                << data.Segments[i].outputLine() << std::endl;
      std::cout << " NewSeg's parent was "
                << seg_p->getParentSeg()->outputLine() << std::endl;
#endif
      SegTouchVec.push_back(&data.Segments[i]);
    }
  }
}

bool Neurogenesis::checkBorders(NeurogenSegment* newSeg) {
  bool rval = false;
  if (NeuronBoundary->isOutsideVolume(newSeg) ||
      _boundingSurfaceMap[newSeg->getParams()->boundingSurface]
          ->isOutsideVolume(newSeg)) {
    rval = true;
  }
  return rval;
}

bool Neurogenesis::checkRadius(NeurogenSegment* newSeg,
                               NeurogenParams* params_p) {
  bool rval = false;
  if (newSeg->getRadius() < params_p->minRadius) {
    rval = true;
  }
  return rval;
}

bool Neurogenesis::checkIntersect2D(int threadID, NeurogenSegment* newSeg) {
  ThreadData& data = _threadData[threadID];
  bool rval = false;
  for (int i = 1; i < data.nrSegments && !rval; i++) {
    if (newSeg == &data.Segments[i]) continue;
    if (newSeg->getParentSeg()->equals(&data.Segments[i])) {
    }
    // std::cout << "Skipping own parent from intersection check" << std::endl;
    else if (newSeg->getParentSeg()->equals(data.Segments[i].getParentSeg())) {
    }
    // std::cout << "Skipping own parent from intersection check" << std::endl;
    else if (newSeg->intersects(&data.Segments[i]))
      rval = true;
  }
  return rval;
}

bool Neurogenesis::checkTouchAndRelax(int threadID, NeurogenSegment* newSeg,
                                      NeurogenParams* params_p) {
  bool rval = false;
  int resampleCount = 0;
  while (touchesAnotherSegWithExceptions(threadID, newSeg, params_p) && !rval) {
    if (resampleCount < params_p->maxResamples) {
#ifdef DBG
      std::cout << "newSeg " << newSeg->outputLine()
                << " touches some other segment!!!!!" << std::endl;
#endif
      newSeg->resampleAfterForces(params_p->gaussSD *
                                  (1.0 + (MAX_RELAX_MULTIPLE - 1.0) *
                                             resampleCount /
                                             params_p->maxResamples));
      newSeg->setLength(
          params_p->umnPerFront /
          pow(2.0 * M_PI * newSeg->getRadius(), params_p->nexpPerFront));
      resampleCount++;
    } else
      rval = true;
  }
  return rval;
}

void Neurogenesis::applyForces(int threadID, NeurogenSegment* newSeg,
                               ShallowArray<NeurogenSegment*>& SegTouchVec,
                               NeurogenParams* params_p) {
  NeurogenSegment* soma = newSeg->getBranch()->getSoma();
  ThreadData& data = _threadData[threadID];
  newSeg->resetBias();
#ifdef DBGFORCES
  std::cout << "Parent: " << newSeg->getParentSeg()->outputLine() << std::endl;
  std::cout << "Segment before Forward Bias: " << std::endl;
  std::cout << newSeg->outputLine() << std::endl;
  std::cout << "Current Segment Bias before Forward Bias: " << std::endl;
  std::cout << newSeg->getBiasX() << "\t" << newSeg->getBiasY() << "\t" <
      newSeg->getBiasZ() << std::endl;
#endif
  newSeg->forwardBias();
#ifdef DBGFORCES
  std::cout << "Current Segment Bias after Forward Bias: " << std::endl;
  std::cout << newSeg->getBiasX() << "\t" << newSeg->getBiasY() << "\t"
            << newSeg->getBiasZ() << std::endl;
#endif
  newSeg->somaRepulsion(soma);

#ifdef DBGFORCES
  std::cout << "Current Segment Bias after somaRepulsion: " << std::endl;
  std::cout << newSeg->getBiasX() << "\t" << newSeg->getBiasY() << "\t"
            << newSeg->getBiasZ() << std::endl;
#endif
  segmentsNear(threadID, SegTouchVec, newSeg, params_p);
  newSeg->homotypicRepulsion(SegTouchVec);
#ifdef DBGFORCES
  std::cout << "Current Segment Bias after homotypic " << std::endl;
  std::cout << newSeg->getBiasX() << "\t" << newSeg->getBiasY() << "\t"
            << newSeg->getBiasZ() << std::endl;
#endif
  newSeg->tissueBoundaryRepulsion(_boundingSurfaceMap);
  newSeg->waypointAttraction();
}

void Neurogenesis::resampleAfterForces(NeurogenSegment* newSeg) {
  double lFront = newSeg->getLength();
  newSeg->resampleAfterForces();
  // std::cout << "Current Segment after all Forces: " << newSeg->outputLine()
  // << std::endl;
  newSeg->setLength(lFront);
}

void Neurogenesis::setSegmentIDs(int threadID, NeurogenParams* params_p,
                                 int& count) {
  ThreadData& data = _threadData[threadID];
  for (unsigned int i = 1; i < data.nrBranches; i++) {
    data.Branches[i].setSegmentIDs(params_p, count);
  }
}

void Neurogenesis::printStats(int threadID, int nid, NeurogenParams* params_p,
                              std::ostream& os, int nrStemSegments, bool names,
                              bool values, const char* namesSeparator,
                              const char* valuesSeparator) {
  ThreadData& data = _threadData[threadID];
  time_t Simrawtime;
  struct tm* Simtimeinfo;
  char timeString[20];
  time(&Simrawtime);
  Simtimeinfo = localtime(&Simrawtime);
  strftime(timeString, 20, "%m-%d-%Y %H:%M", Simtimeinfo);

  if (names) os << "STATS-" << _statsFileName << namesSeparator;
  if (values) os << nid << valuesSeparator;
  if (names) os << "Time" << namesSeparator;
  if (values) os << timeString << valuesSeparator;
  if (names) os << "Soma Surface (from parameters)" << namesSeparator;
  if (values) os << params_p->somaSurface << valuesSeparator;
  if (names) os << "Number of Segments" << namesSeparator;
  if (values) os << data.nrSegments << valuesSeparator;
  if (names) os << "Number of Stems" << namesSeparator;
  if (values) os << params_p->nrStems << valuesSeparator;
  if (names) os << "Number of Stem Segments" << namesSeparator;
  if (values) os << nrStemSegments << valuesSeparator;
  if (names) os << "Number of Bifurcations" << namesSeparator;
  if (values) os << totalBifurcations << valuesSeparator;
  if (names) os << "Number of Branches" << namesSeparator;
  if (values) os << data.nrBranches << valuesSeparator;
  if (names) os << "Overall width" << namesSeparator;
  if (values) os << getSize(threadID, 'X') << valuesSeparator;
  if (names) os << "Overall height" << namesSeparator;
  if (values) os << getSize(threadID, 'Y') << valuesSeparator;
  if (names) os << "Overall depth" << namesSeparator;
  if (values) os << getSize(threadID, 'Z') << valuesSeparator;
  if (names) os << "Average diameter" << namesSeparator;
  if (values) os << getAvgRadius(threadID, params_p) * 2 << valuesSeparator;
  if (names) os << "Total Fiber Length" << namesSeparator;
  if (values) os << getTotalFiberLength(threadID, params_p) << valuesSeparator;
  if (names) os << "Total Surface" << namesSeparator;
  if (values) os << getTotalArea(threadID, params_p) << valuesSeparator;
  if (names) os << "Total Volume" << namesSeparator;
  if (values) os << getTotalVolume(threadID, params_p) << valuesSeparator;
  if (names) os << "Max Branch Order" << namesSeparator;
  if (values) os << getMaxBranchOrder(threadID) << valuesSeparator;
  if (names) os << "Average Rall's Ratio (from parameters)" << namesSeparator;
  if (values) os << params_p->RallsRatio << valuesSeparator;
  if (names) os << "Avg Bifurcation Angle" << namesSeparator;
  if (values) os << getAvgBifurcationAngle(threadID) << valuesSeparator;
  os << std::endl;
}

double Neurogenesis::getAvgRadius(int threadID, NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  double sumRadii = 0;
  for (int i = 1 + params_p->somaSegments; i < data.nrSegments; i++)
    sumRadii = sumRadii + data.Segments[i].getRadius();
  return sumRadii / data.nrSegments;
}

double Neurogenesis::getSize(int threadID, char dimension) {
  ThreadData& data = _threadData[threadID];
  double Size = 0;
  double min = DBL_MAX;
  double max = DBL_MIN;
  for (unsigned int i = 0; i < data.nrSegments; i++) {
    if (dimension == 'X' || dimension == 'x') {
      if (data.Segments[i].getX() < min)
        min = data.Segments[i].getX();
      else if (data.Segments[i].getX() > max)
        max = data.Segments[i].getX();
    } else if (dimension == 'Y' || dimension == 'y') {
      if (data.Segments[i].getY() < min)
        min = data.Segments[i].getY();
      else if (data.Segments[i].getY() > max)
        max = data.Segments[i].getY();
    } else if (dimension == 'Z' || dimension == 'z') {
      if (data.Segments[i].getZ() < min)
        min = data.Segments[i].getZ();
      else if (data.Segments[i].getZ() > max)
        max = data.Segments[i].getZ();
    }
  }
  return max - min;
}

double Neurogenesis::getTotalFiberLength(int threadID,
                                         NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  double FiberLength = 0;
  for (unsigned int i = 1 + params_p->somaSegments; i < data.nrSegments; i++) {
    FiberLength += data.Segments[i].getLength();
  }
  return FiberLength;
}

double Neurogenesis::getTotalArea(int threadID, NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  //NOTE: assume sphere for soma
  double totalArea = 0.0;
  if (data.nrSegments > 0)
  {
    totalArea =
        4.0 * M_PI * data.Segments[0].getRadius() * data.Segments[0].getRadius();
    for (unsigned int i = 1 + params_p->somaSegments; i < data.nrSegments; i++) {
      totalArea += data.Segments[i].getSideArea();
    }

  } 
  return totalArea;
}

double Neurogenesis::getTotalVolume(int threadID, NeurogenParams* params_p) {
  ThreadData& data = _threadData[threadID];
  //NOTE: assume sphere for soma
  double totalVol = 0.0;
  if (data.nrSegments > 0)
  {
    totalVol = 1.3333333333333 * M_PI * data.Segments[0].getRadius() *
      data.Segments[0].getRadius() * data.Segments[0].getRadius();
    for (unsigned int i = 1 + params_p->somaSegments; i < data.nrSegments; i++) {
      totalVol += data.Segments[i].getVolume();
    }
  }
  return totalVol;
}

double Neurogenesis::getAvgBifurcationAngle(int threadID) {
  ThreadData& data = _threadData[threadID];
  double avgAngle = 0;
  double sumAngles = 0;
  unsigned int angleCount = 0;
  for (unsigned int i = 1; i < data.nrBranches; i++) {
    for (int j = i + 1; j < data.nrBranches; j++) {
      if (data.Branches[i].getFirstSegment()->getParent() ==
              data.Branches[j].getFirstSegment()->getParent() &&
          data.Branches[i].getFirstSegment()->getParentSeg()->getType() > 1) {
        sumAngles += data.Branches[i].getFirstSegment()->getAngle(
            data.Branches[j].getFirstSegment());
        angleCount++;
      }
    }
  }
  avgAngle = sumAngles / angleCount;
  return avgAngle * 180.0 / M_PI;
}

int Neurogenesis::getMaxBranchOrder(int threadID) {
  ThreadData& data = _threadData[threadID];
  int maxBranchOrder = 0;
  for (unsigned int i = 0; i < data.nrBranches; i++) {
    int branchOrder = 1;
    NeurogenSegment* parent =
        data.Branches[i].getFirstSegment()->getParentSeg();
    while (parent != 0) {
      parent = parent->getBranch()->getFirstSegment()->getParentSeg();
      branchOrder++;
    }
    if (branchOrder > maxBranchOrder) maxBranchOrder = branchOrder;
  }
  return maxBranchOrder;
}

void Neurogenesis::writeSWC(int threadID, int nid, const char* fileName) {
  ThreadData& data = _threadData[threadID];
  // std::cout << "#Generating SWC file" << std::endl;
  std::ofstream fout;
  if (_somaGenerated[nid])
    fout.open(fileName, std::ios::app);
  else
    fout.open(fileName);
  for (unsigned int i = (_somaGenerated[nid] ? 1 : 0); i < data.nrBranches;
       i++) {
    data.Branches[i].writeToSWC(fout);
  }
  fout.close();
}

// obsolete, only prints out SWC file by Segment rather than by normal branch
// order
// Useful for Debugging purposes

void Neurogenesis::writeSWCbySeg(int threadID, const char* fileName) {
  ThreadData& data = _threadData[threadID];
  // std::cout << "#Generating SWC file by Segs" << std::endl;
  std::ofstream fout;
  fout.open(fileName);
  ;

  for (unsigned int i = 0; i < data.nrSegments; i++) {
    fout << data.Segments[i].outputLine();
    fout << std::endl;
  }
  fout.close();
}

Neurogenesis::~Neurogenesis() {
  delete NeuronBoundary;
  delete[] _threadData;
  for (std::map<std::string, WaypointGenerator*>::iterator iter =
           WaypointGeneratorMap.begin();
       iter != WaypointGeneratorMap.end(); ++iter)
    delete iter->second;
}

Neurogenesis::ThreadData::ThreadData()
    : Segments(0),
      Branches(0),
      nrSegments(0),
      nrStemSegments(0),
      nrBranches(0),
      segmentsSize(0),
      branchesSize(0),
      nid(0) {}

void Neurogenesis::ThreadData::reset(int n, int id, Mutex* mutex) {
  nid = id;
  if (n > segmentsSize) {
#ifndef DISABLE_PTHREADS
    mutex->lock();
#endif
    delete[] Segments;
    Segments = new NeurogenSegment[getBuffAllocationSize(n)];
    segmentsSize = getUsableBuffSize(n);
#ifndef DISABLE_PTHREADS
    mutex->unlock();
#endif
  } else {
    for (int i = 0; i < nrSegments; ++i) Segments[i].reset();
  }
  if (n > branchesSize) {
#ifndef DISABLE_PTHREADS
    mutex->lock();
#endif
    delete[] Branches;
    Branches = new NeurogenBranch[getBuffAllocationSize(n / 2)];
    branchesSize = getUsableBuffSize(n / 2);
#ifndef DISABLE_PTHREADS
    mutex->unlock();
#endif
  } else {
    for (int i = 0; i < nrBranches; ++i) Branches[i].reset();
  }
  nrSegments = nrBranches = 0;
}

NeurogenSegment* Neurogenesis::ThreadData::allocSegment(NeurogenSegment& seg) {
  NeurogenSegment* rval = &Segments[nrSegments];
  ++nrSegments;
  rval->set(seg);
  return rval;
}

NeurogenSegment* Neurogenesis::ThreadData::allocSegment(
    NeurogenParams* params) {
  NeurogenSegment* rval = &Segments[nrSegments];
  ++nrSegments;
  rval->set(params);
  return rval;
}

NeurogenSegment* Neurogenesis::ThreadData::allocSegment(
    int id, int type, double x, double y, double z, double r, int p,
    NeurogenParams* params) {
  NeurogenSegment* rval = &Segments[nrSegments];
  ++nrSegments;
  rval->set(id, type, x, y, z, r, p, params);
  return rval;
}

NeurogenSegment* Neurogenesis::ThreadData::allocSegment(
    int type, double x, double y, double z, double r, NeurogenParams* params) {
  NeurogenSegment* rval = &Segments[nrSegments];
  ++nrSegments;
  rval->set(type, x, y, z, r, params);
  return rval;
}

NeurogenBranch* Neurogenesis::ThreadData::allocBranch() {
  NeurogenBranch* rval = &Branches[nrBranches];
  ++nrBranches;
  return rval;
}

void Neurogenesis::ThreadData::deleteSegment() { --nrSegments; }

void Neurogenesis::ThreadData::deleteBranch() {
  --nrBranches;
  Branches[nrBranches].reset();
}

Neurogenesis::ThreadData::~ThreadData() {
  delete[] Segments;
  delete[] Branches;
}
