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

#include "CommandLine.h"
#include "Utility.h"
#include "Parser.h"

#include <string>
#include <iostream>

CommandLine::CommandLine() 
   : _fileName(""), _static(false)
{
}

bool CommandLine::parse(int argc, char** argv) 
{
   int width = 75;

   Parser parser;

   parser.addOption(Option('f', "mdlFile", Option::TYPE_NONE));
   parser.addOption(Option('i', "includePath", Option::TYPE_REQUIRED));
   parser.addOption(Option('s', "static", Option::TYPE_NONE));
   parser.addOption(Option('n', "no-warning", Option::TYPE_NONE));

   // No arguments: show help and quit
   if (argc == 1) {
      parser.help();
      return false;
   }

   Parser::ParameterVector parameterVector = parser.parse(argc, argv);
   try {
      for (Parser::ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
         Option option = parameterVector[i].getOption();
         Parser::String value = parameterVector[i].getValue();
         if (option == Option::OPTION_NONE) {
            std::cout << "The input MDL File Name is " << value << "\n";
            _fileName = value;
         } else if (option.getShortName() == 's') {
            std::cout << "Static link will be applied\n";
            _static = true;
         } else if (option.getShortName() == 'n') {
            std::cout << "No warning is printed\n";
            _printWarning = false;
         } else if (option.getShortName() == 'i') {
            mdl::tokenize(value, _includePath, ':');
         }
      }
   } catch (Parser::Exception exception) {
      std::cerr << "Exception: " << exception.getMessage() << "...exiting...\n";
   }

   return true;
}
