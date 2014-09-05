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
