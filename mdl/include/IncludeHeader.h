// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IncludeHeader_H
#define IncludeHeader_H
#include "Mdl.h"

#include <string>
#include "MacroConditional.h"

// This class generates #includes
class IncludeHeader
{
   public:
      IncludeHeader(const std::string& name = "", 
		    const std::string& conditional = "");
      ~IncludeHeader() {};

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

      std::string getHeaderCode() const;

      bool operator <(const IncludeHeader& other) const {
	 return (_name + _macroConditional.getName()) 
	    < (other._name + _macroConditional.getName());
      }

   private:
      std::string _name;

      MacroConditional _macroConditional;
};

#endif
