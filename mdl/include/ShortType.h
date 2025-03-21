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

#ifndef ShortType_H
#define ShortType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "SignedType.h"

class ShortType : public SignedType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~ShortType();        

      virtual std::string getDescriptor() const;
      virtual std::string getCapitalDescriptor() const;
};

#endif // ShortType_H
