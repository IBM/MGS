#include <memory>
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

#ifndef CopyConstructorMethod_H
#define CopyConstructorMethod_H
#include "Mdl.h"

#include "ConstructorMethod.h"
#include <string>
#include <vector>

class Attribute;

class CopyConstructorMethod : public ConstructorMethod
{
   public:
      CopyConstructorMethod();
      CopyConstructorMethod(const std::string& name, 
			    const std::string& returnStr = "",
			    const std::string& functionBody = "",
			    const std::string& initializationStr = "");
      virtual void duplicate(std::unique_ptr<Method>&& dup) const;
      virtual ~CopyConstructorMethod();
      void addCopyConstructorInitializers(
	 const std::vector<Attribute*>& attributes,
	 const std::string& beginning,
	 const std::string& copyFrom);
   protected:
      virtual void callInitMethod(
	 const std::vector<Attribute*>::const_iterator& it,
	 std::string& initStr, const std::string& copyFrom);
};

#endif
