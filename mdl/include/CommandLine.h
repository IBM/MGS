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
      std::vector<std::string> _includePath;
};


#endif // CommandLine_H
