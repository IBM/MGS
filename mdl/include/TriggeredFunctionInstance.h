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

#ifndef TriggeredFunctionInstance_H
#define TriggeredFunctionInstance_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "TriggeredFunction.h"

class Generatable;

class TriggeredFunctionInstance : public TriggeredFunction {

   public:
      TriggeredFunctionInstance(const std::string& name, RunType runType);
      virtual void duplicate(
	 std::auto_ptr<TriggeredFunction>& rv) const;
      virtual ~TriggeredFunctionInstance();
      
   protected:
      virtual std::string getTriggerableType (
	 const std::string& modelName) const;     
      virtual std::string getTab() const {
	 return "\t";
      }

};

#endif // TriggeredFunctionInstance_H
