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

#ifndef C_array_H
#define C_array_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class DataType;
class ArrayType;

class C_array : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_array();
      C_array(C_typeClassifier* tc);
      C_array(const C_array& rv);
      virtual void duplicate(std::auto_ptr<C_array>& rv) const;
      void releaseDataType(std::auto_ptr<DataType>& dt);
      virtual ~C_array();

   private:
      C_typeClassifier* _typeClassifier;
      ArrayType* _arrayType;
};


#endif // C_array_H
