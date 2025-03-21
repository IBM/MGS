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

#ifndef EdgeSetType_H
#define EdgeSetType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class EdgeSetType : public LensType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~EdgeSetType();        

      virtual std::string getDescriptor() const;

      // This method returns if the pointer of the specific dataType is
      // meant to be owned by the class.
      virtual bool shouldBeOwned() const;
};

#endif // EdgeSetType_H
