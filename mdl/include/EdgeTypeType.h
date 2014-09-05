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

#ifndef EdgeTypeType_H
#define EdgeTypeType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class EdgeTypeType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~EdgeTypeType();        

      virtual std::string getDescriptor() const;
};

#endif // EdgeTypeType_H
