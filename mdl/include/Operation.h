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

#ifndef Operation_H
#define Operation_H
#include "Mdl.h"

#include <memory>
#include <string>

class Predicate;

class Operation {

   public:
      enum _Type {_String, _Bool, _General, _Invalid};
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~Operation();
      virtual void operate(Predicate* p1, Predicate* p2, Predicate* cur);

   protected:
      _Type classify(std::string s);

};


#endif // Operation_H
