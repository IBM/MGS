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

#ifndef C_typeCore_H
#define C_typeCore_H
#include "Mdl.h"

#include "DataType.h"
#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;

class C_typeCore : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_typeCore();
      C_typeCore(DataType* dt);
      C_typeCore(const std::string& s);
      C_typeCore(const C_typeCore& rv);
      virtual void duplicate(std::auto_ptr<C_typeCore>& rv) const;
      void releaseDataType(std::auto_ptr<DataType>& dt);
      virtual ~C_typeCore();
      
   private:
      DataType* _dataType;
      std::string _id;

};


#endif // C_typeCore_H
