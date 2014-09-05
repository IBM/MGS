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

#ifndef DeepPointerArray_H
#define DeepPointerArray_H
#include "Copyright.h"

#include "Array.h"

template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE, 
	  unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class DeepPointerArray : public Array<T*>
{
   public:
      DeepPointerArray();
      DeepPointerArray(const DeepPointerArray* rv);
      DeepPointerArray(const DeepPointerArray& rv);
      DeepPointerArray& operator=(const DeepPointerArray& rv);
      virtual void duplicate(std::auto_ptr<Array<T*> >& rv) const;
      virtual void duplicate(std::auto_ptr<DeepPointerArray<T, 
			     blockSize, blockIncrementSize> >& rv) const;
      virtual ~DeepPointerArray();

      virtual void clear() {
	 destructContents();
	 Array<T*>::clear();
      }

   protected:
      virtual void internalCopy(T*& lval, T*& rval);
      virtual unsigned getBlockSize() const {
	 return blockSize;
      }
      virtual unsigned getBlockIncrementSize() const {
	 return blockIncrementSize;
      }
      void destructContents();
      void copyContents(const DeepPointerArray& rv);
};

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DeepPointerArray<T, blockSize, blockIncrementSize>::DeepPointerArray()
   : Array<T*>(blockIncrementSize)
{
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DeepPointerArray<T, blockSize, blockIncrementSize>::DeepPointerArray(
   const DeepPointerArray* rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array<T*>::copyContents(*rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DeepPointerArray<T, blockSize, blockIncrementSize>::DeepPointerArray(
   const DeepPointerArray& rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array<T*>::copyContents(rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DeepPointerArray<T, blockSize, blockIncrementSize>& 
DeepPointerArray<T, blockSize, blockIncrementSize>::operator=(
   const DeepPointerArray& rv)
{
   if (this == &rv) {
      return *this;
   }
   // !!!! Important destruct before op=
   destructContents();
   Array<T*>::operator=(rv);
   return *this;
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::duplicate(
   std::auto_ptr<Array<T*> >& rv) const
{
   rv.reset(new DeepPointerArray<T, blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::duplicate(
   std::auto_ptr<
   DeepPointerArray<T, blockSize, blockIncrementSize> >& rv) const
{
   rv.reset(new DeepPointerArray<T, blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DeepPointerArray<T, blockSize, blockIncrementSize>::~DeepPointerArray()
{
   destructContents();
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::internalCopy(
   T*& lval, T*& rval)
{
   T* retVal = new T();
   *retVal = *rval;
   lval = retVal;
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::destructContents()
{
   unsigned index = 0;
   for (unsigned i = 0; (i < this->_activeBlocks) && (index < this->_size); 
	i++) {
      for (unsigned j = 0; (j < blockSize) && (index < this->_size); j++) {
	 index++;
	 delete this->_blocksArray[i][j];
      }
   }
}

#endif
