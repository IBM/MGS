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
      void setNegateCondition() {
	 _negate_condition = true;
      }
      void unsetNegateCondition() {
	 _negate_condition = false;
      }

      std::string getBeginning() const;
      std::string getEnding() const;

      void addExtraTest(const std::string& name) {
	 _extraTest = name;
      }
   private:
      std::string _name;
      bool _negate_condition; //if True, then print '#if ! defined(...)'
      std::string _extraTest;
      //std::vector<std::string> _extraTest;
};

#endif
