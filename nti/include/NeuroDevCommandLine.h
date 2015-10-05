// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef NeuroDevCommandLine_H
#define NeuroDevCommandLine_H

#include "NeuroDevParser.h"
#include <vector>
#include <string>

class NeuroDevCommandLine {

   public:
      NeuroDevCommandLine();
      ~NeuroDevCommandLine() {}
      bool parse(int argc, char** argv);
      bool parse(const char*);
      bool parse(std::vector<std::string>& argv);

      std::string getInputFileName() {return _inputFileName;}
      std::string getOutputFileName() {return _outputFileName;}
      void outToIn() {_inputFileName=_outputFileName;}
      std::string getBinaryFileName() {return _binaryFileName;}
      int getNumberOfThreads() {return _nThreads;}
      int getNumberOfSlicers() {return _nSlicers;}
      int getNumberOfDetectors() {return _nDetectors;}
      int getMaxIterations() {return _maxIterations;}
      int getCapsPerCpt() {return _capsPerCpt;}
      double getPointSpacing() {return _pointSpacingFactor;}
      std::string getExperimentName() {return _experimentName;}
      int getX() {return _x;}
      int getY() {return _y;}
      int getZ() {return _z;}
      int getInitialFront() {return _initialFront>0 ? _initialFront : 1;}
      std::string getParamFileName() {return _paramFileName;}
      double getEnergyCon() {return _energyCon;}
      double getTimeStep() {return _timeStep;}
      bool getResample() {return _resample;}
      bool getClientConnect() {return _clientConnect;}
      double getAppositionSamplingRate() {return _appositionRate;}
      std::string getDecomposition() {return _decomposition;}
      std::string getOutputFormat() {return _outputFormat;}
      long getSeed() {return _seed;}
      bool getCellBodyMigration() {return _cellBodyMigration;}
   private:
      void addOptions(NeuroDevParser& parser);
   private:
      std::string _inputFileName;
      std::string _outputFileName;
      std::string _binaryFileName;
      int _nThreads;
      int _nSlicers;
      int _nDetectors;
      int _maxIterations;
      int _capsPerCpt;
      double _pointSpacingFactor;
      std::string _experimentName;
      int _x, _y, _z;
      int _initialFront;
      std::string _paramFileName;      
      double _energyCon;
      double _timeStep;
      double _appositionRate;
      bool _resample;
      bool _clientConnect;
      bool _verbose;
      std::string _decomposition;
      std::string _outputFormat;
      long _seed;
      bool _cellBodyMigration;
};


#endif // NeuroDevCommandLine_H
