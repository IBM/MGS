// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef MacroConditional_H
#define MacroConditional_H
#include "Mdl.h"

#include <string>

// This class is used by code generators. If the name is anything else
// than "", the code is dependent upon that that macro being defined
// so appropriate macros are created for beginning and ending for the 
// C++ preprocessor.
class MacroConditional
{
   public:
      MacroConditional(const std::string& name = "");
      ~MacroConditional() {};

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name) {
	 _name = name;
      }

      std::string getBeginning() const;
      std::string getEnding() const;

   private:
      std::string _name;
};

#endif
