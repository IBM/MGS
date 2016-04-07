// =================================================================
//
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

#include "NeuroDevCommandLine.h"
#include "NeuroDevParser.h"
#include "ConstantData.h"  // Tuan, added Sept-15-2015
#include "StringUtils.h"
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>

#define DEFAULT_ENERGY_CON 0.5
#define DEFAULT_TIME_STEP 0.01

NeuroDevCommandLine::NeuroDevCommandLine()
    : _inputFileName(""),
      _outputFileName(""),
      _binaryFileName(""),
      _nThreads(1),
      _nSlicers(0),
      _nDetectors(0),
      _maxIterations(0),
      _capsPerCpt(1),
      _pointSpacingFactor(1.0),
      _experimentName(""),
      _x(-1),
      _y(-1),
      _z(-1),
      _initialFront(0),
      _paramFileName(""),
      _energyCon(DEFAULT_ENERGY_CON),
      _timeStep(DEFAULT_TIME_STEP),
      _appositionRate(1.0),
      _resample(false),
      _clientConnect(false),
      _verbose(false),
      _decomposition("volume"),
      _outputFormat(""),
      _seed(12345678),
      _cellBodyMigration(false)
{
#ifdef SILENT_MODE
  _verbose = false;
#endif
}

void NeuroDevCommandLine::addOptions(NeuroDevParser& parser)
{
  parser.addOption(NeuroDevParser::Option(
      'a', "apposition-sampling-rate", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'b', "binary-input-file", NeuroDevParser::Option::TYPE_OPTIONAL));
  parser.addOption(NeuroDevParser::Option('c', "client-connect",
                                          NeuroDevParser::Option::TYPE_NONE));
  parser.addOption(NeuroDevParser::Option(
      'd', "n-detectors", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'e', "energy-crit", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'f', "initial-front", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(
      NeuroDevParser::Option('g', "geometric-resampling-factor",
                             NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(
      NeuroDevParser::Option('h', "help", NeuroDevParser::Option::TYPE_NONE));
  parser.addOption(NeuroDevParser::Option(
      'i', "text-input-file", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'j', "threads", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'k', "cell-body-migration", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'm', "max-iterations", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'n', "decomposition", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'o', "output-file", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'p', "param-file", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'q', "seed", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(
      NeuroDevParser::Option('r', "compartment-resampling-factor",
                             NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      's', "n-slicers", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      't', "time-step", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'u', "output-format", NeuroDevParser::Option::TYPE_OPTIONAL));
  parser.addOption(NeuroDevParser::Option(
      'v', "experiment-name", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'x', "slicing-geom-x", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'y', "slicing-geom-y", NeuroDevParser::Option::TYPE_REQUIRED));
  parser.addOption(NeuroDevParser::Option(
      'z', "slicing-geom-z", NeuroDevParser::Option::TYPE_REQUIRED));
}

bool NeuroDevCommandLine::parse(std::vector<std::string>& argv)
{
  NeuroDevParser parser;
  this->addOptions(parser);
  // No arguments: show help and quit
  int argc = argv.size();
  if (argc == 1)
  {
    parser.help();
    return false;
  }

  NeuroDevParser::ParameterVector parameterVector = parser.parse(argv);
  try
  {
    for (NeuroDevParser::ParameterVector::size_type i = 0;
         i < parameterVector.size(); i++)
    {
      NeuroDevParser::Option option = parameterVector[i].getOption();
      NeuroDevParser::String value = parameterVector[i].getValue();
      if (option.getShortName() == 'a')
      {
        if (_verbose) std::cout << "Apposition sampling rate set.\n";
        _appositionRate = atof(value.c_str());
      }
      else if (option.getShortName() == 'b')
      {
        if (_verbose) std::cout << "Binary output filename set.\n";
        _binaryFileName = value;
      }
      else if (option.getShortName() == 'c')
      {
        if (_verbose) std::cout << "Client connect set.\n";
        _clientConnect = true;
      }
      else if (option.getShortName() == 'd')
      {
        if (_verbose)
          std::cout << "Number of detectors set : " << value << ".\n";
        _nDetectors = atoi(value.c_str());
      }
      else if (option.getShortName() == 'e')
      {
        if (_verbose)
          std::cout << "Energy convergence criterion set : " << value << ".\n";
        _energyCon = atof(value.c_str());
      }
      else if (option.getShortName() == 'f')
      {
        if (_verbose) std::cout << "Initial front set : " << value << ".\n";
        _initialFront = atoi(value.c_str());
      }
      else if (option.getShortName() == 'g')
      {
        if (_verbose)
          std::cout << "Geometric resampling factor set : " << value << ".\n";
        _resample = true;
        _pointSpacingFactor = atof(value.c_str());
      }
      else if (option.getShortName() == 'h')
      {
        std::cout << "Help:\n";
        parser.help();
        return false;
      }
      else if (option == NeuroDevParser::Option::ND_OPTION_NONE ||
               option.getShortName() == 'i')
      {
        if (_verbose) std::cout << "Input filename set : " << value << ".\n";
        _inputFileName = value;
      }
      else if (option.getShortName() == 'j')
      {
        if (_verbose) std::cout << "Number of threads set : " << value << ".\n";
        _nThreads = atoi(value.c_str());
      }
      else if (option.getShortName() == 'k')
      {
        if (_verbose) std::cout << "Cell body migration set.\n";
        _cellBodyMigration = true;
      }
      else if (option.getShortName() == 'm')
      {
        if (_verbose)
          std::cout << "Maximum iterations set : " << value << ".\n";
        _maxIterations = atoi(value.c_str());
      }
      else if (option.getShortName() == 'n')
      {
        if (_verbose) std::cout << "Decomposition set : " << value << ".\n";
        _decomposition = value;
      }
      else if (option.getShortName() == 'o')
      {
        if (_verbose)
          std::cout << "Text output filename set : " << value << ".\n";
        _outputFileName = value;
      }
      else if (option.getShortName() == 'p')
      {
        if (_verbose)
          std::cout << "Parameter filename set : " << value << ".\n";
        _paramFileName = value;
      }
      else if (option.getShortName() == 'q')
      {
        if (_verbose) std::cout << "Seed set : " << value << ".\n";
        _seed = atol(value.c_str());
      }
      else if (option.getShortName() == 'r')
      {
        if (_verbose)
          std::cout << "Compartment resampling factor set : " << value << ".\n";
        _resample = true;
        _capsPerCpt = atoi(value.c_str());
      }
      else if (option.getShortName() == 's')
      {
        if (_verbose) std::cout << "Number of slicers set : " << value << ".\n";
        _nSlicers = atoi(value.c_str());
      }
      else if (option.getShortName() == 't')
      {
        if (_verbose) std::cout << "Time step set.\n";
        _timeStep = atof(value.c_str());
      }
      else if (option.getShortName() == 'u')
      {
        if (_verbose) std::cout << "Output format set.\n";
        _outputFormat = value;
        if (_outputFormat == "tb") _outputFormat = "bt";
      }
      else if (option.getShortName() == 'v')
      {
        if (_verbose) std::cout << "Experiment name set.\n";
        _experimentName = value;
      }
      else if (option.getShortName() == 'x')
      {
        if (_verbose)
          std::cout << "X slicing geometry set : " << value << ".\n";
        _x = atoi(value.c_str());
      }
      else if (option.getShortName() == 'y')
      {
        if (_verbose)
          std::cout << "Y slicing geometry set : " << value << ".\n";
        _y = atoi(value.c_str());
      }
      else if (option.getShortName() == 'z')
      {
        if (_verbose)
          std::cout << "Z slicing geometry set : " << value << ".\n";
        _z = atoi(value.c_str());
      }
    }
  }
  catch (NeuroDevParser::Exception exception)
  {
    std::cerr << "Exception: " << exception.getMessage() << "...exiting...\n";
    exit(0);
  }

  if (_verbose)
  {
    std::cout << "\nSettings for the simulation: \n"
              << "\t"
              << "appositionSamplingRate = " << _appositionRate << "\n"
              << "\t"
              << "binary input file name = " << _binaryFileName << "\n"
              << "\t"
              << "clientConnect = " << (_clientConnect ? "true" : "false")
              << "\n"
              << "\t"
              << "n detectors = " << _nDetectors << "\n"
              << "\t"
              << "energy criterion = " << _energyCon << "\n"
              << "\t"
              << "initial front = " << _initialFront << "\n"
              << "\t"
              << "geometric resampling = " << _pointSpacingFactor << "\n"
              << "\t"
              << "text input file = " << _inputFileName << "\n"
              << "\t"
              << "n threads = " << _nThreads << "\n"
              << "\t"
              << "cell body migration = "
              << (_cellBodyMigration ? "true" : "false") << "\n"
              << "\t"
              << "max iterations = " << _maxIterations << "\n"
              << "\t"
              << "decomposition = " << _decomposition << "\n"
              << "\t"
              << "output file name = " << _outputFileName << "\n"
              << "\t"
              << "parameter file = " << _paramFileName << "\n"
              << "\t"
              << "seed = " << _seed << "\n"
              << "\t"
              << "compartment resampling = " << _capsPerCpt << "\n"
              << "\t"
              << "n slicers = " << _nSlicers << "\n"
              << "\t"
              << "time step = " << _timeStep << "\n"
              << "\t"
              << "outputFormat = " << _outputFormat << "\n"
              << "\t"
              << "experimentName = " << _experimentName << "\n"
              << "\t"
              << "x (slicing geometry) = " << _x << "\n"
              << "\t"
              << "y (slicing geometry) = " << _y << "\n"
              << "\t"
              << "z (slicing geometry) = " << _z << "\n"
              << "\t"
              << "resample = " << (_resample ? "true" : "false") << "\n"
              << std::endl;
  }
  return true;
}

bool NeuroDevCommandLine::parse(int argc, char** argv)
{
  NeuroDevParser parser;

  this->addOptions(parser);
  // No arguments: show help and quit
  if (argc == 1)
  {
    parser.help();
    return false;
  }

  NeuroDevParser::ParameterVector parameterVector = parser.parse(argc, argv);
  try
  {
    for (NeuroDevParser::ParameterVector::size_type i = 0;
         i < parameterVector.size(); i++)
    {
      NeuroDevParser::Option option = parameterVector[i].getOption();
      NeuroDevParser::String value = parameterVector[i].getValue();
      if (option.getShortName() == 'a')
      {
        if (_verbose) std::cout << "Apposition sampling rate set.\n";
        _appositionRate = atof(value.c_str());
      }
      else if (option.getShortName() == 'b')
      {
        if (_verbose) std::cout << "Binary output filename set.\n";
        _binaryFileName = value;
      }
      else if (option.getShortName() == 'c')
      {
        if (_verbose) std::cout << "Client connect set.\n";
        _clientConnect = true;
      }
      else if (option.getShortName() == 'd')
      {
        if (_verbose)
          std::cout << "Number of detectors set : " << value << ".\n";
        _nDetectors = atoi(value.c_str());
      }
      else if (option.getShortName() == 'e')
      {
        if (_verbose)
          std::cout << "Energy convergence criterion set : " << value << ".\n";
        _energyCon = atof(value.c_str());
      }
      else if (option.getShortName() == 'f')
      {
        if (_verbose) std::cout << "Initial front set : " << value << ".\n";
        _initialFront = atoi(value.c_str());
      }
      else if (option.getShortName() == 'g')
      {
        if (_verbose)
          std::cout << "Geometric resampling factor set : " << value << ".\n";
        _resample = true;
        _pointSpacingFactor = atof(value.c_str());
      }
      else if (option.getShortName() == 'h')
      {
        std::cout << "Help:\n";
        parser.help();
        return false;
      }
      else if (option == NeuroDevParser::Option::ND_OPTION_NONE ||
               option.getShortName() == 'i')
      {
        if (_verbose) std::cout << "Input filename set : " << value << ".\n";
        _inputFileName = value;
      }
      else if (option.getShortName() == 'j')
      {
        if (_verbose) std::cout << "Number of threads set : " << value << ".\n";
        _nThreads = atoi(value.c_str());
      }
      else if (option.getShortName() == 'k')
      {
        if (_verbose) std::cout << "Cell body migration set.\n";
        _cellBodyMigration = true;
      }
      else if (option.getShortName() == 'm')
      {
        if (_verbose)
          std::cout << "Maximum iterations set : " << value << ".\n";
        _maxIterations = atoi(value.c_str());
      }
      else if (option.getShortName() == 'n')
      {
        if (_verbose) std::cout << "Decomposition set : " << value << ".\n";
        _decomposition = value;
      }
      else if (option.getShortName() == 'o')
      {
        if (_verbose)
          std::cout << "Text output filename set : " << value << ".\n";
        _outputFileName = value;
      }
      else if (option.getShortName() == 'p')
      {
        if (_verbose)
          std::cout << "Parameter filename set : " << value << ".\n";
        _paramFileName = value;
      }
      else if (option.getShortName() == 'q')
      {
        if (_verbose) std::cout << "Seed set : " << value << ".\n";
        _seed = atol(value.c_str());
      }
      else if (option.getShortName() == 'r')
      {
        if (_verbose)
          std::cout << "Compartment resampling factor set : " << value << ".\n";
        _resample = true;
        _capsPerCpt = atoi(value.c_str());
      }
      else if (option.getShortName() == 's')
      {
        if (_verbose) std::cout << "Number of slicers set : " << value << ".\n";
        _nSlicers = atoi(value.c_str());
      }
      else if (option.getShortName() == 't')
      {
        if (_verbose) std::cout << "Time step set.\n";
        _timeStep = atof(value.c_str());
      }
      else if (option.getShortName() == 'u')
      {
        if (_verbose) std::cout << "Output format set.\n";
        _outputFormat = value;
        if (_outputFormat == "tb") _outputFormat = "bt";
      }
      else if (option.getShortName() == 'v')
      {
        if (_verbose) std::cout << "Experiment name set.\n";
        _experimentName = value;
      }
      else if (option.getShortName() == 'x')
      {
        if (_verbose)
          std::cout << "X slicing geometry set : " << value << ".\n";
        _x = atoi(value.c_str());
      }
      else if (option.getShortName() == 'y')
      {
        if (_verbose)
          std::cout << "Y slicing geometry set : " << value << ".\n";
        _y = atoi(value.c_str());
      }
      else if (option.getShortName() == 'z')
      {
        if (_verbose)
          std::cout << "Z slicing geometry set : " << value << ".\n";
        _z = atoi(value.c_str());
      }
    }
  }
  catch (NeuroDevParser::Exception exception)
  {
    std::cerr << "Exception: " << exception.getMessage() << "...exiting...\n";
    exit(0);
  }

  if (_verbose)
  {
    std::cout << "\nSettings for the simulation: \n"
              << "\t"
              << "appositionSamplingRate = " << _appositionRate << "\n"
              << "\t"
              << "binary input file name = " << _binaryFileName << "\n"
              << "\t"
              << "clientConnect = " << (_clientConnect ? "true" : "false")
              << "\n"
              << "\t"
              << "n detectors = " << _nDetectors << "\n"
              << "\t"
              << "energy criterion = " << _energyCon << "\n"
              << "\t"
              << "initial front = " << _initialFront << "\n"
              << "\t"
              << "geometric resampling = " << _pointSpacingFactor << "\n"
              << "\t"
              << "text input file = " << _inputFileName << "\n"
              << "\t"
              << "n threads = " << _nThreads << "\n"
              << "\t"
              << "cell body migration = "
              << (_cellBodyMigration ? "true" : "false") << "\n"
              << "\t"
              << "max iterations = " << _maxIterations << "\n"
              << "\t"
              << "decomposition = " << _decomposition << "\n"
              << "\t"
              << "output file name = " << _outputFileName << "\n"
              << "\t"
              << "parameter file = " << _paramFileName << "\n"
              << "\t"
              << "seed = " << _seed << "\n"
              << "\t"
              << "compartment resampling = " << _capsPerCpt << "\n"
              << "\t"
              << "n slicers = " << _nSlicers << "\n"
              << "\t"
              << "time step = " << _timeStep << "\n"
              << "\t"
              << "outputFormat = " << _outputFormat << "\n"
              << "\t"
              << "experimentName = " << _experimentName << "\n"
              << "\t"
              << "x (slicing geometry) = " << _x << "\n"
              << "\t"
              << "y (slicing geometry) = " << _y << "\n"
              << "\t"
              << "z (slicing geometry) = " << _z << "\n"
              << "\t"
              << "resample = " << (_resample ? "true" : "false") << "\n"
              << std::endl;
  }
  return true;
}

bool NeuroDevCommandLine::parse(const char* line)
{
  int argc = 0;
  char* argv[ConstantData::MaxNumCmdLineArgs];
  /*char commandLine[1024];
std::strcpy(commandLine, line);
argv[argc]=strtok(commandLine, " ");
*/
  std::string commandLine(line);
  std::vector<std::string> tokens;

  StringUtils::Tokenize(commandLine, tokens, " ");

  if (tokens.size() >= ConstantData::MaxNumCmdLineArgs)
  {
    std::cerr << "NeuroDevCommandLine: Too many arguments! (limits to : "
              << ConstantData::MaxNumCmdLineArgs << ")" << std::endl;
    exit(0);
  }
  /*  argv[argc]=strtok(commandLine.c_str(), " ");
while (argv[argc] != 0) {
  if (++argc==ConstantData::MaxNumCmdLineArgs) {
    std::cerr<<"NeuroDevCommandLine: Too many arguments! (limits to : " <<
ConstantData::MaxNumCmdLineArgs << ")" <<std::endl;
    exit(0);
  }
  argv[argc]=strtok(0, " ");
}

return parse(argc, argv);
*/
  return parse(tokens);
}
