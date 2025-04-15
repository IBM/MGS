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

#ifndef NEUROGENPARAMS_H_
#define NEUROGENPARAMS_H_

#include "rndm.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

// For printing out debugging output during Neuron Generation

#define _SILENT_
#ifdef VERBOSE
#undef _SILENT_
#endif

#ifndef _SILENT_
#define DBG
#define DBGFORCES
#define SHOWTERMINATIONS
#endif

#define MAX_RELAX_MULTIPLE 2.0 // used in Neurogenesis::checkTouchAndRelax

class NeurogenParams
{
 public:
  NeurogenParams(int rank);
  NeurogenParams(std::string fileName, int rank);
  void load(std::string fileName, int rank);
  void set_Params();
  void printParams();
  void printParams(std::ostream& os,
		   bool names, bool values, 
		   const char* preamble, 
		   const char* namesSeparator, const char* valuesSeparator);
  //void setDefaults();

  double getGaussian(double mu, double sigma);

  // Starting Parameters
  double startX;
  double startY;
  double startZ;
  int nrStems;
  double somaSurface;
  double genZ;
  int somaSegments;

  // Growth and branching parameters
  double startRadius;
  double umnPerFront;
  double nexpPerFront;
  double radiusRate;
  double minRadius;
  double RallsRatio;
  double branchDiameterAsymmetry;
  double branchProb;
  double minBifurcationAngle;
  double maxBifurcationAngle;
  double minInitialStemAngle;
  double somaRepulsion;
  double somaDistanceExp;
  double homotypicRepulsion;
  double homotypicDistanceExp;
  double boundaryRepulsion;
  double boundaryDistanceExp;
  double intolerance;
  double forwardBias;
  std::string waypointGenerator;
  double waypointAttraction;
  double waypointDistanceExp;
  double waypointExtent;


  /// Global
  double maxFiberLength;
  double width;
  double height;
  double depth;
  int maxBifurcations;
  unsigned int maxSegments;

  // Number
  long RandSeed;
  double gaussSD;
  double maxResamples;
  std::string boundingSurface;
  std::string terminalField;

  int _rank;
  RNG _rng;

 private:
  void convertDegreesToRadians();
};



#endif /* PARAMS_H_ */
