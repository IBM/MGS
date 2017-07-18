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

// Created by Heraldo Memelli
// summer 2012

#ifndef NEUROGENESIS_H
#define NEUROGENESIS_H

#include <math.h>
#include "ShallowArray.h"
#include "NeurogenSegment.h"
#include "NeurogenBranch.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "rndm.h"

class NeurogenBranch;
class NeurogenSegment;
class BoundingVolume;
class NeurogenParams;
class Mutex;
class BoundingSurfaceMesh;
class WaypointGenerator;
class Mutex;

class Neurogenesis {
  private:
  class ThreadData {
public:
    ThreadData();
    void reset(int n, int id, Mutex* mutex);
    NeurogenSegment* allocSegment(NeurogenSegment& seg);
    NeurogenSegment* allocSegment(NeurogenParams* params);
    NeurogenSegment* allocSegment(int id, int type, double x, double y,
                                  double z, double r, int p,
                                  NeurogenParams* params);
    NeurogenSegment* allocSegment(int type, double x, double y, double z,
                                  double r, NeurogenParams* params);
    NeurogenBranch* allocBranch();
    void deleteSegment();
    void deleteBranch();
    ~ThreadData();

    NeurogenSegment* Segments;
    NeurogenBranch* Branches;
    int nrSegments, nrBranches;
    int segmentsSize, branchesSize;
    int nrStemSegments;
    int nid;
  };

  public:
  Neurogenesis(int rank, int size, int nThreads, std::string statsFileName,
               std::string parsFileName, bool stdout, bool fout, int branchType,
               std::map<std::string, BoundingSurfaceMesh*>& boundingSurfaceMap);
  ~Neurogenesis();

  // void run(int neuronBegin, int nNeurons, NeurogenParams** params_p, char**
  // fileNames, bool* somaGenerated);
  void run(int neuronBegin, int nNeurons, NeurogenParams** params_p,
           std::vector<std::string>& fileNames, bool* somaGenerated);
  void doWork(int threadID, int i, ThreadData* data, Mutex* mutex);
  void generateArbors(int threadID, int nid, int& nrStems);
  void readSoma(int threadID, int nid, int& count, NeurogenParams* params_p);
  void generateSoma(int threadID, NeurogenParams* params_p, int& count);
  void generateStems(int threadID, NeurogenParams* params_p, int nid,
                     int& nrStems);
  void extendNeurites(int threadID, NeurogenParams* params_p);
  void applyForces(int threadID, NeurogenSegment* newSeg,
                   ShallowArray<NeurogenSegment*>& SegTouchVec,
                   NeurogenParams* params_p);
  void resampleAfterForces(NeurogenSegment* newSeg);
  void setSegmentIDs(int threadID, NeurogenParams* params_p, int& count);

  // output SWC.
  void writeSWC(int threadID, int nid, const char* fileName);
  ;
  void writeSWCbySeg(int threadID,
                     const char* fileName);  // Debug purposes only
  void printStats(int threadID, int nid, NeurogenParams* params_p,
                  std::ostream& os, int nrStemSegments, bool names, bool values,
                  const char* namesSeparator, const char* valuesSeparator);

  private:
  // Important Neurogeneration functions
  bool branchWithAngle(int threadID, NeurogenSegment* seg_p,
                       NeurogenParams* params_p);
  bool growSegment(int threadID, NeurogenSegment* seg_p,
                   NeurogenParams* params_p);

  // used for checking for overlaps
  bool touchesAnotherSeg(int threadID, NeurogenSegment* seg_p,
                         NeurogenParams* params_p);
  bool touchesAnotherSegWithExceptions(int threadID, NeurogenSegment* seg_p,
                                       NeurogenParams* params_p);
  bool withinMinimumStemAngleOfAnotherSeg(int threadID, NeurogenSegment* seg_p,
                                          double relaxationFactor,
                                          NeurogenParams* params_p);
  bool checkIntersect2D(int threadID, NeurogenSegment* newSeg);
  void segmentsNear(int threadID, ShallowArray<NeurogenSegment*>& SegTouchVec,
                    NeurogenSegment* seg_p, NeurogenParams* params_p);
  bool checkBorders(NeurogenSegment* seg_p);
  bool checkRadius(NeurogenSegment* seg_p, NeurogenParams* params_p);
  bool checkTouchAndRelax(int threadID, NeurogenSegment* seg_p,
                          NeurogenParams* params_p);

  // Generate Stats functions
  double getSize(int threadID, char dimension);
  double getAvgRadius(int threadID, NeurogenParams* params_p);
  double getTotalArea(int threadID, NeurogenParams* params_p);
  double getTotalVolume(int threadID, NeurogenParams* params_p);
  double getAvgBifurcationAngle(int threadID);
  double getTotalFiberLength(int threadID, NeurogenParams* params_p);
  int getMaxBranchOrder(int threadID);

  BoundingVolume* NeuronBoundary;
  int totalBifurcations;
  int _neuronBegin;
  NeurogenParams** _neurogenParams;
  std::vector<std::string> _neurogenFileNames;
  std::string _statsFileName, _parsFileName;
  bool _stdout, _fout;
  bool* _somaGenerated;
  std::ostringstream _opars, _ostats;
  int _branchType;
  std::map<std::string, BoundingSurfaceMesh*>& _boundingSurfaceMap;
  std::map<std::string, WaypointGenerator*> WaypointGeneratorMap;

  int _rank;
  int _size;
  int _nThreads;
  ThreadData* _threadData;
};

#endif /* NEUROGEN_H_ */

