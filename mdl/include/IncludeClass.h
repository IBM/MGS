// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IncludeClass_H
#define IncludeClass_H
#include "Mdl.h"

#include <string>
#include "MacroConditional.h"

// This class generates #includes
class IncludeClass
{
   public:
      IncludeClass(const std::string& name = "", 
		   const std::string& conditional = "");
      ~IncludeClass() {};

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name) {
	 _name = name;
      }

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      

      bool operator <(const IncludeClass& other) const {
	 return (_name + _macroConditional.getName()) 
	    < (other._name + _macroConditional.getName());
      }

      std::string getClassCode() const;
      std::string getHeaderCode() const;

   private:
      std::string _name;

      MacroConditional _macroConditional;
};

#endif
