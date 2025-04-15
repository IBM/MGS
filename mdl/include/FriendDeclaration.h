// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FriendDeclaration_H
#define FriendDeclaration_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "MacroConditional.h"

class Class;

class FriendDeclaration {

   public:
      FriendDeclaration(const std::string& name, 
			const std::string& conditional = "");
      ~FriendDeclaration();

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      
     
      std::string getCodeString() const;

   protected:
      std::string _name;
      MacroConditional _macroConditional;
};


#endif // FriendDeclaration_H
