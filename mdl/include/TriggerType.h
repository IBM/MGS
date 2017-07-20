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

#ifndef TriggerType_H
#define TriggerType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class TriggerType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~TriggerType();        

      virtual std::string getDescriptor() const;
};

#endif // TriggerType_H
