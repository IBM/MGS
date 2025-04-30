// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "CommandLine.h"
#include "Utility.h"
#include "Parser.h"

#include <string>
#include <iostream>

CommandLine::CommandLine() 
   : _fileName(""), _static(false), _skipIncludes(true)
{
}

CommandLine::CommandLine(CommandLine& cl) 
   : _fileName(cl._fileName), _static(cl._static), 
   _printWarning(cl._printWarning),
   _skipIncludes(cl._skipIncludes),
   _includePath(cl._includePath)
{
}

bool CommandLine::parse(int argc, char** argv) 
{
   int width = 75;

   Parser parser;

   parser.addOption(Option('f', "mdl-file", Option::TYPE_NONE));
   parser.addOption(Option('i', "include-path", Option::TYPE_REQUIRED));
   parser.addOption(Option('s', "static-linking", Option::TYPE_NONE));
   parser.addOption(Option('n', "no-warnings", Option::TYPE_NONE));
   parser.addOption(Option('k', "keep-includes", Option::TYPE_NONE));

   // No arguments: show help and quit
   if (argc == 1) {
      parser.help();
      return false;
   }

   Parser::ParameterVector parameterVector = parser.parse(argc, argv);
   try {
      for (Parser::ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
         Option option = parameterVector[i].getOption();
         Parser::CustomString value = parameterVector[i].getValue();
         if (option == Option::OPTION_NONE) {
            std::cout << "Input MDL filename : " << value << "\n";
            _fileName = value;
         } else if (option.getShortName() == 's') {
            std::cout << "-s: Static linking applied.\n";
            _static = true;
         } else if (option.getShortName() == 'n') {
            std::cout << "-n: Warnings suppressed.\n";
            _printWarning = false;
         } else if (option.getShortName() == 'i') {
            mdl::tokenize(value, _includePath, ':');
            std::cout<<"-i: Include path set to: "<<value<<".\n";
         } else if (option.getShortName() == 'k') {
            _skipIncludes = false;
            std::cout<<"-k: Keeping included files current by regenerating.\n";
         }
      }
   } catch (Parser::Exception exception) {
      std::cerr << "ERROR: " << exception.getMessage() << "...Exiting...\n";
   }

   return true;
}
