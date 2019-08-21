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

/*
 * neuroGen.cxx : main method for Neuro Generation project
 *
 */

#include <mpi.h>

#include "NeurogenParams.h"
#include "Neurogenesis.h"
#include "CompositeSwc.h"
#include "BoundingSurfaceMesh.h"
#include "rndm.h"

//#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <float.h>
#define DBG

//#define DBG

#define PAR_FILE_INDEX 8
#define N_BRANCH_TYPES 3

void print_help() {
  std::cerr << "USAGE:" << std::endl;
  std::cerr << std::endl;
  std::cerr << "./neuroGen [-tissue tissuefile] [-par parameterfilename_1 parameterfilename_2 parameterfilename_3] "
               "[-stdout] [-n]" << std::endl;
  std::cerr << "Example1: ./neuroGen -tissue \"minicolumn.txt\" " << std::endl;
  std::cerr << "Example2: (no tissue file, just three parameter filenames for one "
               "neuron): ./neuroGen -par \"params_1.txt params_2.txt params_3.txt\"" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef DBG
  std::cerr << "Hello from neuroGen: " << argv[0] << std::endl;
#endif
  int rank, size;

  MPI_Init(&argc, &argv);  // Initialize MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Barrier(MPI_COMM_WORLD);

  double start, now;
  start = MPI_Wtime();
  bool* somaGenerated = 0;
  std::string baseParFileName = "NULL", swcFileName = "NULL";
  std::vector<double> complexities;
  int neuronBegin = 0, neuronEnd = 0;
  RNG rng;

#ifdef DBG
  std::cerr << "Entering branch generation... " << argv[0] << std::endl;
#endif

  for (int branchType = 0; branchType < N_BRANCH_TYPES;
       ++branchType) {  // 0:axon, 1:basal, 2:apical
    std::string btype;
    if (branchType == 0)
      btype = "axons";
    else if (branchType == 1)
      btype = "denda";
    else if (branchType == 2)
      btype = "dendb";
    std::map<std::string, BoundingSurfaceMesh*> boundingSurfaceMap;
    bool parFile = false;
    bool btissueFile = false;
    std::string tissueFileName;
    bool stdout = false;
    bool fout = false;
    double composite = 0.0;
    int nthreads = 1;
    int N = 1;
    if (argc == 1) {
      print_help();
      MPI_Finalize();
      exit(0);
    }
    for (int i = 1; i < argc; i++) {
      std::string sCurrentArg = argv[i];
#ifdef DBG
      std::cerr<<sCurrentArg<<std::endl;
#endif      
      if ((sCurrentArg == "-?") || (sCurrentArg == "-help") ||
          (sCurrentArg == "-info")) {
        print_help();
        MPI_Finalize();
        exit(0);
      }
      if ((sCurrentArg == "-par")) {
	assert(argc>=i+3);
	baseParFileName = argv[i + 1 + branchType];
        for (int bt = 1; bt <= N_BRANCH_TYPES; ++bt) {
          // the last par file from axon, denda, dendb, will be the base name
          // from the new swc file, contcatenating all outputs
	  if (std::string(argv[i+bt]) != "NULL") {
          //if (strcmp(argv[i + bt], "NULL")) {
            swcFileName = argv[i + bt];
            break;
          }
        }
	parFile = false;
        if (baseParFileName != "NULL") parFile = true;
#ifndef _SILENT_
        std::cerr << "Parameter file name: " << baseParFileName << std::endl;
#endif
	i += N_BRANCH_TYPES;
      } else if ((sCurrentArg == "-tissue")) {
        btissueFile = true;
        tissueFileName = argv[i + 1];
#ifndef _SILENT_
        std::cerr << "Tissue file name: " << tissueFileName << std::endl;
#endif
        if (branchType == 0) {
          // only need to do the following once per neuron
          double totalComplexity = 0.0;
          std::ifstream tissueFile(tissueFileName.c_str());
          while (tissueFile.good()) {
            std::string line;
            getline(tissueFile, line);
            if (line != "" && line.at(0) != '#') {
              std::string str = line;
              std::stringstream strstr(str);
              std::istream_iterator<std::string> it(strstr);
              std::istream_iterator<std::string> end;
              std::vector<std::string> results(it, end);
              if (results.size() >= PAR_FILE_INDEX + N_BRANCH_TYPES) {
                // use first paramfile in tissue to seed RNG below
                for (int bt = 0; bt < N_BRANCH_TYPES; ++bt) {
                  if (results.at(PAR_FILE_INDEX + bt) != "NULL") {
                    baseParFileName = results.at(PAR_FILE_INDEX + bt);
                    break;
                  }
                }
                double complexity = 0.0;
                std::ifstream testFile(results.at(0).c_str());
                if (!testFile) {
                  if (results.size() > PAR_FILE_INDEX + N_BRANCH_TYPES) {
                    complexity = atof(
                        results.at(PAR_FILE_INDEX + N_BRANCH_TYPES).c_str());
                  } else
                    complexity = 1.0;
                } else
                  testFile.close();
                totalComplexity += complexity;
                complexities.push_back(complexity);
              }
            }
          }
          tissueFile.close();
          double targetComplexity = totalComplexity / double(size);
          double runningComplexity = 0.0;
          int count = 0, divisor = size;
          int nNeurons = complexities.size();
          somaGenerated = new bool[nNeurons];
          bool assigned = false;
          for (int i = 0; i < nNeurons; ++i) {
            somaGenerated[i] = false;
            if ((runningComplexity += complexities[i]) >= targetComplexity) {
              --divisor;
              neuronBegin = neuronEnd;
              if (neuronBegin == i ||
                  runningComplexity - targetComplexity <
                      targetComplexity -
                          (runningComplexity - complexities[i])) {
                totalComplexity -= runningComplexity;
                targetComplexity = totalComplexity / divisor;
                runningComplexity = 0.0;
                neuronEnd = i + 1;
              } else {
                totalComplexity -= (runningComplexity - complexities[i]);
                targetComplexity = totalComplexity / divisor;
                runningComplexity = complexities[i];
                neuronEnd = i;
              }
              if (count == rank) {
                assigned = true;
                break;
              }
              ++count;
            }
          }
          if (!assigned) neuronBegin = neuronEnd;
        }
      } else if ((sCurrentArg == "-stdout")) {
        stdout = true;
      } else if ((sCurrentArg == "-fout")) {
        fout = true;
      } else if ((sCurrentArg == "-composite")) {
        if (btissueFile)
          composite = atof(argv[i + 1]);
        else
          std::cerr << "Warning : composite file creation supported only for "
                       "tissue file input!" << std::endl;
        i++;
      } else if ((sCurrentArg == "-n")) {
        N = atoi(argv[i + 1]);
        i++;
      } else if ((sCurrentArg == "-j")) {
        nthreads = atoi(argv[i + 1]);
        i++;
      }
    }
    std::string statsFileName, parsFileName, compositeSwcFileName;
    int nNeuronsGenerated = 0;
    NeurogenParams** params = 0;
    std::vector<std::string> fileNames;
    // char** fileNames=0;
    /*
    if (branchType == 0) {
      NeurogenParams params_p(baseParFileName, rank);
      rng.reSeed(lrandom(params_p._rng), rank);
    }
    */
    if (!btissueFile) {  // if there is no tissue file but just a parameter file
                         // to create only 1 neuron.
      if (parFile) {
        int ln = baseParFileName.length();
        std::string statsFileName(baseParFileName);
        std::string parsFileName(baseParFileName);
        statsFileName.erase(ln - 4, 4);
        parsFileName.erase(ln - 4, 4);
        if (swcFileName != "NULL")
          swcFileName.erase(swcFileName.length() - 4, 4);

        std::ostringstream statsFileNameStream, parsFileNameStream;
        statsFileNameStream << statsFileName << "." << btype << ".out";
        statsFileName = statsFileNameStream.str();
        parsFileNameStream << parsFileName << "." << btype << ".par";
        parsFileName = parsFileNameStream.str();

        int neuronBegin = int(floor(double(rank) * double(N) / double(size)));
        int neuronEnd =
            int(floor(double(rank + 1) * double(N) / double(size))) - 1;
        nNeuronsGenerated = neuronEnd - neuronBegin + 1;
        if (nNeuronsGenerated > 0) {
          params = new NeurogenParams* [nNeuronsGenerated];
          // fileNames = new char* [nNeuronsGenerated];
          for (int i = 0; i < nNeuronsGenerated; ++i) params[i] = 0;
          if (somaGenerated == 0) {
            somaGenerated = new bool[nNeuronsGenerated];
            for (int i = 0; i < nNeuronsGenerated; ++i)
              somaGenerated[i] = false;
          }
        }
        int idx = 0;
        for (int nid = neuronBegin; nid <= neuronEnd; ++nid, ++idx) {
          std::ostringstream filename;
          filename << swcFileName << "_" << nid << ".swc";
          /*
                fileNames[idx] = new char[filename.str().length()];
          strcpy(fileNames[idx], filename.str().c_str());
                  */
          fileNames.push_back(filename.str());
          /*
           * the rand_seed provided via the Params file is the seed for the RNG which generates the 
           *            random seeds for each neuron's params[idx]
           *  initially, all params[idx] get this rand_seed, but then each is updated with new seed value
           */
          if (parFile) {
            params[idx] = new NeurogenParams(baseParFileName, rank);
	    if (idx==0) rng.reSeedShared(params[idx]->RandSeed);
            params[idx]->RandSeed = lrandom(rng);
            params[idx]->_rng.reSeedShared(params[idx]->RandSeed);
          }
        }
        if (nNeuronsGenerated > 0)
          boundingSurfaceMap[params[0]->boundingSurface] =
              new BoundingSurfaceMesh(params[0]->boundingSurface);
        Neurogenesis NG(rank, size, nthreads, statsFileName, parsFileName,
                        stdout, fout, branchType + 2, boundingSurfaceMap);
        NG.run(neuronBegin, nNeuronsGenerated, params, fileNames,
               somaGenerated);
      }
    } else {  // there is a tissue file
      nNeuronsGenerated = neuronEnd - neuronBegin;
      params = new NeurogenParams* [nNeuronsGenerated];
      // fileNames = new char* [nNeuronsGenerated];
      for (int i = 0; i < nNeuronsGenerated; ++i) params[i] = 0;

      int ln = tissueFileName.length();
      std::string statsFileName(tissueFileName);
      std::string parsFileName(tissueFileName);
      statsFileName.erase(ln - 4, 4);
      parsFileName.erase(ln - 4, 4);

      std::ostringstream statsFileNameStream, parsFileNameStream;
      statsFileNameStream << statsFileName << "." << btype << ".out";
      statsFileName = statsFileNameStream.str();
      parsFileNameStream << parsFileName << "." << btype << ".par";
      parsFileName = parsFileNameStream.str();

      if (composite > 0) {
        compositeSwcFileName = tissueFileName;
        compositeSwcFileName.erase(ln - 4, 4);
        std::ostringstream compositeSwcFileNameStream;
        compositeSwcFileNameStream << compositeSwcFileName << "." << rank
                                   << ".swc";
        compositeSwcFileName = compositeSwcFileNameStream.str();
      }

      int neuronID = 0, idx = 0;
      std::ifstream tissueFile(tissueFileName.c_str());
      if (tissueFile.is_open()) {
        while (tissueFile.good()) {
          std::string line;
          getline(tissueFile, line);
          if (line != "" && line.at(0) != '#') {
            if (neuronID >= neuronBegin && neuronID < neuronEnd) {
              std::string str = line;
              // construct a stream from the string
              std::stringstream strstr(str);

              // use stream iterators to copy the stream to the vector as
              // whitespace separated strings
              std::istream_iterator<std::string> it(strstr);
              std::istream_iterator<std::string> end;
              std::vector<std::string> results(it, end);
#ifdef DBG
              for (int j = 0; j < results.size(); j++)
                std::cerr << results[j] << std::endl;
#endif
              /*
  fileNames[idx] = new char[results.at(0).length()];
  strcpy(fileNames[idx], results.at(0).c_str());
              */
              fileNames.push_back(results.at(0));
              if (complexities[idx] > 0 &&
                  results.at(PAR_FILE_INDEX + branchType) != "NULL") {
                params[idx] = new NeurogenParams(
                    results.at(PAR_FILE_INDEX + branchType), rank);
                params[idx]->RandSeed = lrandom(rng);
                params[idx]->_rng.reSeedShared(params[idx]->RandSeed);
                params[idx]->startX = atof(results.at(4).c_str());
                params[idx]->startY = atof(results.at(5).c_str());
                params[idx]->startZ = atof(results.at(6).c_str());
                std::map<std::string, BoundingSurfaceMesh*>::iterator miter =
                    boundingSurfaceMap.find(params[idx]->boundingSurface);
                if (miter == boundingSurfaceMap.end())
                  boundingSurfaceMap[params[idx]->boundingSurface] =
                      new BoundingSurfaceMesh(params[idx]->boundingSurface);
              }
              ++idx;
            }
            neuronID++;
          }
        }
      }
      tissueFile.clear();
      tissueFile.close();

      Neurogenesis NG(rank, size, nthreads, statsFileName, parsFileName, stdout,
                      fout, branchType + 2, boundingSurfaceMap);
      NG.run(neuronBegin, nNeuronsGenerated, params, fileNames, somaGenerated);
    }

    if (composite > 0 && rank == 0)
      CompositeSwc(tissueFileName.c_str(), compositeSwcFileName.c_str(),
                   composite, false);

    for (int nid = 0; nid < nNeuronsGenerated; ++nid) {
      // delete[] fileNames[nid];
      delete params[nid];
    }
    // delete[] fileNames;
    delete[] params;
    std::map<std::string, BoundingSurfaceMesh*>::iterator miter,
        mend = boundingSurfaceMap.end();
    for (miter = boundingSurfaceMap.begin(); miter != mend; ++miter)
      delete miter->second;
  }
  now = MPI_Wtime();
  if (rank == 0) std::cerr << "Compute Time : " << now - start << std::endl;

  MPI_Finalize();
}
