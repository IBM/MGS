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

#ifndef ArrayType_H
#define ArrayType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include <vector>
#include "DataType.h"

class ArrayType : public DataType {
   public:
      ArrayType();
      ArrayType(DataType* dt);
      ArrayType(const ArrayType& rv);
      ArrayType& operator=(const ArrayType& rv);
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~ArrayType();        

      const DataType* getType() const {
	 return _type;
      }
      void setType(std::auto_ptr<DataType>& type) {
	 delete _type;
	 _type = type.release();	 
      }

      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;
      virtual std::string getHeaderDataItemString() const;
      virtual std::string getInitializerString(
	 const std::string& diArg, int level = 0, 
	 bool isIterator = true, bool forPSet = false) const;

      // This function returns a string showing what kind of array should
      // be produced. Shallow, Deep, Duplicate
      std::string getPrefixArrayType() const;
      virtual std::string getDataItemString() const;
      virtual std::string getArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level) const;

      // This function returns if the pointer of the specific dataType is meant
      // to be owned by the class.
      virtual bool shouldBeOwned() const;

      // Checks the _type's if they would require anything to copy recursively.
      virtual bool anythingToCopy();

      virtual bool isArray() const {
	 return true;
      }

      virtual bool isSuitableForInterface() const;

      // This function will add this dataType to a Class as a proxy attribute.
      virtual void addProxyAttribute(std::auto_ptr<Class>& instance) const;

      // This function sets the characteristics of the array container.
      virtual void setArrayCharacteristics(
	 unsigned blockSize, unsigned incrementSize);
      virtual void getSubStructDescriptors(std::set<std::string>& subStructTypes) const;

   protected:
      virtual std::string getDescriptor() const;

   private:
      void copyOwnedHeap(const ArrayType& rv);
      void destructOwnedHeap();
      bool _static;
      DataType* _type;
      unsigned _blockSize;
      unsigned _incrementSize;
};

#endif // ArrayType_H
