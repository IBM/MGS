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

#ifndef STRUCTTYPE_H
#define STRUCTTYPE_H
#include "Copyright.h"

#include "InstanceFactory.h"
#include <memory>
#include <vector>
#include <list>
#include <string>

class Struct;
class DataItem;
class InstanceFactoryQueriable;
class NDPairList;

class StructType : public InstanceFactory
{
   public:
      StructType();
      StructType(const StructType& rv);
      StructType& operator=(const StructType& rv);
      virtual void duplicate(std::auto_ptr<StructType>& dup) const=0;
      virtual void getStruct(std::auto_ptr<Struct> & r_aptr)=0;
      virtual Struct* getStruct()=0;
      virtual std::string getName()=0;
      virtual std::string getDescription()=0;
      virtual ~StructType();
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c);
   protected:
      std::list<Struct*> _structList;      
   private:
      void copyContents(const StructType& rv);
      void destructContents();
};
#endif
