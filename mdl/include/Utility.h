// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Utility_H
#define Utility_H
#include "Mdl.h"

class Class;
class DataType;

#include <string>
#include <vector>
#include "Phase.h"
#include "TriggeredFunction.h"

#include "MemberContainer.h"

namespace mdl {

   void stripNameForCG(std::string& name);

   bool findInPhases(const std::string& name, 
		     const std::vector<Phase*>& vec);

   bool findInTriggeredFunctions(const std::string& name, 
				 const std::vector<TriggeredFunction*>& vec);

   void tokenize(const std::string& data, std::vector<std::string>& tokens, 
		 char separator);

   void addOptionalServicesToClass(Class& instance, 
				   MemberContainer<DataType>& services);

}

#endif
