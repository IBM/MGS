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

#ifndef CONSTANTTYPE_H
#define CONSTANTTYPE_H
//#include "Copyright.h"

#include "InstanceFactory.h"
#include <memory>
#include <vector>
#include <string>
#include "Constant.h"
#include "DuplicatePointerArray.h"

//class Constant;
class DataItem;
class InstanceFactoryQueriable;
class NDPairList;

class ConstantType : public InstanceFactory
{
   public:
      ConstantType();
      virtual void duplicate(std::auto_ptr<ConstantType>& dup) const=0;
      virtual void getConstant(std::auto_ptr<Constant> & r_aptr)=0;
      virtual Constant* getConstant()=0;
      virtual std::string getName()=0;
      virtual std::string getDescription()=0;
      virtual ~ConstantType();
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c);
   protected:
      DuplicatePointerArray<Constant> _constantList;
};
#endif
