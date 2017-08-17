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

#ifndef LongType_H
#define LongType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "SignedType.h"

class LongType : public SignedType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~LongType();        

      virtual std::string getDescriptor() const;
      virtual std::string getCapitalDescriptor() const;
};
#endif // LongType_H
