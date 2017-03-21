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

#ifndef PredicateFunction_H
#define PredicateFunction_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "Class.h"

class ConnectionCCBase;

class PredicateFunction {

   public:
      PredicateFunction(const std::string& name);
      virtual void duplicate(std::auto_ptr<PredicateFunction>& rv) const;
      virtual ~PredicateFunction();
     
      std::string getName() const {
	 return _name;
      }

      void generateInstanceMethod(Class& instance, bool pureVirtual,
				  const ConnectionCCBase& ccBase) const;

      std::string getString() const;

   protected:
      std::string _name;
};


#endif // PredicateFunction _H
