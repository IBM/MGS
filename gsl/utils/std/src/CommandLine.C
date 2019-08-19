// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//   
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "CommandLine.h"
#include "Parser.h"
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <sstream>

CommandLine::CommandLine(bool verbose)
  : 
  
#ifndef DISABLE_PTHREADS
  
  _threads(0), _userInterface("none"), _guiPort(4000), _bindCpus(false),
  
#endif // DISABLE_PTHREADS
  _seed(12345), // default seed
  _gslFile(""), _enableErd(false), _numWorkUnits(0), _readGraph(false), _outputGraph(false), _simulate(true), _verbose(verbose)
, _gpuID(-1) 
{
#ifdef SILENT_MODE
  _verbose=false;
#endif
}


bool CommandLine::parse(int argc, char** argv)
{
  Parser parser;

  parser.addOption(Parser::Option('h', "help", Parser::Option::TYPE_NONE));
  parser.addOption(Parser::Option('f', "gslfile", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('e', "enableErd", Parser::Option::TYPE_NONE));

#ifndef DISABLE_PTHREADS

  parser.addOption(Parser::Option('t', "threads", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('u', "userInt", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('p', "guiport", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('w', "workunits", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('b', "bindCpus", Parser::Option::TYPE_NONE));

#endif // DISABLE_PTHREADS
  parser.addOption(Parser::Option('d', "deviceID", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('s', "seedRng", Parser::Option::TYPE_REQUIRED));
  parser.addOption(Parser::Option('r', "readGraph", Parser::Option::TYPE_NONE));
  parser.addOption(Parser::Option('o', "outputGraph", Parser::Option::TYPE_NONE));
  parser.addOption(Parser::Option('m', "suppressSimulation", Parser::Option::TYPE_NONE));

   // No arguments: show help and quit
  if (argc == 1) {
    parser.help();
    return false;
  }

  try {
    Parser::ParameterVector parameterVector = parser.parse(argc, argv);
    if (_verbose && parameterVector.size()>0) std::cout << std::endl;
    for (Parser::ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
      Parser::Option option = parameterVector[i].getOption();
      Parser::String value = parameterVector[i].getValue();
      if (option.getShortName() == 'h') {
	std::cout << "Help:\n";
	parser.help();
	return false;
      } else if (option == Parser::Option::OPTION_NONE || option.getShortName() == 'f') {
	if (_verbose) std::cout << "Filename set : " << value << ".\n";
	_gslFile = value;
      } else if (option.getShortName() == 'e') {
	if (_verbose) std::cout << "Enable edge relational data was set.\n";
	_enableErd = true;
#ifndef DISABLE_PTHREADS
      } else if (option.getShortName() == 't') {
	if (_verbose) std::cout << "Number of threads set : " << value << ".\n";
	_threads = atoi(value.c_str());
      } else if (option.getShortName() == 'u') {
	if (_verbose) std::cout << "User interface set : " << value << ".\n";
	_userInterface = value.c_str();
      } else if (option.getShortName() == 'p') {
	if (_verbose) std::cout << "GUI port number set : " << value << ".\n";
	_guiPort = atoi(value.c_str());
      } else if (option.getShortName() == 'w') {
	if (_verbose) std::cout << "Number of workUnits set : " << value << ".\n";
	_numWorkUnits = atoi(value.c_str());
      } else if (option.getShortName() == 'b') {
	if (_verbose) std::cout << "Bind CPUs was set.\n";
	_bindCpus = true;
#endif
      } else if (option.getShortName() == 's') {
	if (_verbose) std::cout << "Seed was set by user.\n";
	_seed = strtoul(value.c_str(), NULL, 0);
        if (_seed < 1)
          {
            std::cerr << "Exception: seed must be >= 1 ...exiting...\n";
            exit(0);
          }
      } else if (option.getShortName() == 'r') {
	if (_verbose) std::cout << "Read graph partitioning was set.\n";
	_readGraph = true;
      } else if (option.getShortName() == 'o') {
	if (_verbose) std::cout << "Output graph partitioning was set.\n";
	_outputGraph = true;
      } else if (option.getShortName() == 'm') {
	if (_verbose) std::cout << "Suppress simulation was set.\n";
	_simulate = false;
      } else if (option.getShortName() == 'd') {
	if (_verbose) std::cout << "The GPU deviceID to use .\n";
	_gpuID = atoi(value.c_str());
      }
    }
#ifdef HAVE_GPU
    if (_verbose) std::cout << "Utilizing the GPU.\n";
#endif // HAVE_GPU          
  } catch (Parser::Exception exception) {
    std::cerr << "Exception: " << exception.getMessage() << "...exiting...\n";
    return false;
    //exit(0);
  }
   
#ifndef DISABLE_PTHREADS
  if (_threads == 0) _threads = 1;
#endif
   
  if (!_numWorkUnits) {
#ifndef DISABLE_PTHREADS
    _numWorkUnits = _threads;
#else
    _numWorkUnits = 1;
#endif
  }

  if (_verbose) {
    std::cout << "\nSettings for the simulation: \n" 
	      << "\t" <<"gslFile = " << _gslFile << "\n"
	      << "\t" <<"enableErd = " << (_enableErd ? "true" : "false") << "\n"
      
#ifndef DISABLE_PTHREADS
      
	      << "\t" <<"threads = " << _threads << "\n"
	      << "\t" <<"userInt = " << _userInterface << "\n"
	      << "\t" <<"guiport = " << _guiPort << "\n"
	      << "\t" <<"bindCpus = " << (_bindCpus ? "true" : "false") << "\n"
	      << "\t" <<"workUnits = " << _numWorkUnits << "\n"
      
#endif // DISABLE_PTHREADS
      
	      << "\t" <<"RNGseed = " << _seed << "\n"
	      << "\t" <<"readGraph = " << (_readGraph ? "true" : "false") << "\n"
	      << "\t" <<"outputGraph = " << (_outputGraph ? "true" : "false") << "\n"
	      << "\t" <<"suppressSimulation = " << (_simulate ? "false" : "true") << "\n"
	      << std::endl;
  }
  return true;
}

bool CommandLine::convertToInt(const std::string& str, int& num)
{
  int len = str.size();
  const char* data = str.data();
  for (int i = 0; i < len; ++i) {
    if (isdigit(data[i]) == 0) {
      return false;
    }
  }
  num = atoi(data);
  return true;
}


CommandLine::~CommandLine()
{
}
