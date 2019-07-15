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
#include <cassert>
#include <vector>
#include <string>
#include <iostream>

// This class is used by code generators. If the name is anything else
// than "", the code is dependent upon that that macro being defined
// so appropriate macros are created for beginning and ending for the 
// C++ preprocessor.
// CHANGE:
//  Jul-07-2019: (Tuan M. HoangTrong) add supports for multiple macros
//   e.g. #if defined(MACRO_01) and defined(MACRO_02)
#define USE_COMPLEX_MACRO 1

class MacroConditional
{
   public:
      MacroConditional(const std::string& name = "");
#if defined(USE_COMPLEX_MACRO)
      MacroConditional(const std::vector<std::string>& names);
#endif
      ~MacroConditional() {};

#if defined(USE_COMPLEX_MACRO)
      const std::string& getName() const 
      {
	 assert(_names.size()==1);
	 return _names[0];
      }
#else
      const std::string& getName() const 
      {
	 return _name;
      }
#endif

#if defined(USE_COMPLEX_MACRO)
      void setName(const std::string& name) 
      {
	 //_name = name;
	 _names.clear();
	 _names.push_back(name);
      }
#else
      void setName(const std::string& name) 
      {
	 _name = name;
      }
#endif
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
#if defined(USE_COMPLEX_MACRO)
      std::vector<std::string> _names;
#else
      std::string _name;
#endif
      bool _negate_condition; //if True, then print '#if ! defined(...)'
      std::string _extraTest;
      //std::vector<std::string> _extraTest;
};

#endif
