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
