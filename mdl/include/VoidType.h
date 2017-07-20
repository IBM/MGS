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

#ifndef VoidType_H
#define VoidType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class VoidType : public DataType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~VoidType();        

      virtual std::string getDescriptor() const;
};

#endif // VoidType_h


