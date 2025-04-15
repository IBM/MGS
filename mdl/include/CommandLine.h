// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CommandLine_H
#define CommandLine_H
#include "Mdl.h"

#include <vector>
#include <string>

class CommandLine {

   public:
      CommandLine();
      CommandLine(CommandLine& cl);
      bool parse(int argc, char** argv);

      const std::string& getFileName() {
	      return _fileName;
      }
      bool isStatic() const {
	      return _static;
      }
      bool skipIncludes() const {
	      return _skipIncludes;
      }
      bool printWarning() const {
   	   return _printWarning;
      }
      const std::vector<std::string>& getIncludePath() {
	      return _includePath;
      }

   private:
      void clear();
      std::string _fileName;
      bool _static;
      bool _printWarning=true;
      bool _skipIncludes;
      std::vector<std::string> _includePath;
};


#endif // CommandLine_H
