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
// =================================================================

#ifndef CommandLine_H
#define CommandLine_H
#include "Mdl.h"

#include <vector>
#include <string>

class CommandLine {

   public:
      CommandLine();
      bool parse(int argc, char** argv);

      const std::string& getFileName() {
	 return _fileName;
      }

      bool isStatic() const {
	 return _static;
      }	 

      const std::vector<std::string>& getIncludePath() {
	 return _includePath;
      }

   private:
      void clear();
      std::string _fileName;
      bool _static;
      std::vector<std::string> _includePath;
};


#endif // CommandLine_H
