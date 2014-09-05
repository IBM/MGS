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

#ifndef NDPairListType_H
#define NDPairListType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class NDPairListType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~NDPairListType();        

      virtual std::string getDescriptor() const;

      // This method returns if the pointer of the specific dataType
      // is meant to be owned by the class.
      virtual bool shouldBeOwned() const;
};

#endif // NDPairListType_H
