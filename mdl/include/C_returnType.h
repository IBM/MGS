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

#ifndef C_returnType_H
#define C_returnType_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class DataType;

class C_returnType : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_returnType(bool v = false);
      C_returnType(C_typeClassifier* _t);
      C_returnType(const C_returnType& rv);
      virtual void duplicate(std::auto_ptr<C_returnType>& rv) const;
      void releaseDataType(std::auto_ptr<DataType>& dt);
      virtual ~C_returnType();

      bool isVoid() const;
      C_typeClassifier* getType() const;

   private:
      bool _void;
      C_typeClassifier* _type;
      DataType* _dataType;
};


#endif // C_returnType_H
