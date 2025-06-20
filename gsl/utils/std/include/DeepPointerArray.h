// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DeepPointerArray_H
#define DeepPointerArray_H
#include "Copyright.h"

#include "Array.h"

/* IMPORTANT
 * In GPU scenario: blockSize, blockIncrementSize do NOT play any role
 */
#if defined(HAVE_GPU) 
  #include "DeepPointerArray_GPU.h"
#endif

#ifdef USE_FLATARRAY_FOR_CONVENTIONAL_ARRAY
//Only C++11
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE,
          unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
using DeepPointerArray = DeepPointerArray_Flat<T, 0, blockIncrementSize>;
#else
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE, 
	  unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class DeepPointerArray : public Array<T*>
{
   public:
      DeepPointerArray();
      DeepPointerArray(const DeepPointerArray* rv);
      DeepPointerArray(const DeepPointerArray& rv);
      DeepPointerArray& operator=(const DeepPointerArray& rv);
      virtual void duplicate(std::unique_ptr<Array<T*> >& rv) const;
      virtual void duplicate(std::unique_ptr<DeepPointerArray<T, 
			     blockSize, blockIncrementSize> >& rv) const;
      virtual ~DeepPointerArray();

      virtual void clear() {
	 destructContents();
	 Array<T*>::clear();
      }

   protected:
      virtual void internalCopy(T*& lval, T*& rval);
//#if ! (defined(HAVE_GPU) )
      virtual unsigned getBlockSize() const {
	 return blockSize;
      }
      virtual unsigned getBlockIncrementSize() const {
	 return blockIncrementSize;
      }
//#endif
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
   std::unique_ptr<Array<T*> >& rv) const
{
   rv.reset(new DeepPointerArray<T, blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::duplicate(
   std::unique_ptr<
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
   /* for any class 'T' that we want to be on Unfied Memory
    * we should make such class derived from 'Managed' class
    * in that we overwrite the new and delete operator
    */
   T* retVal = new T();
   *retVal = *rval;
   lval = retVal;
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DeepPointerArray<T, blockSize, blockIncrementSize>::destructContents()
{
//#if defined(HAVE_GPU) 
//   for (unsigned j = 0; j < this->_size; j++) {
//      //TUAN TODO FIX
//      //use this->_mem_location == MemLocation::CPU  or MemLocation::UnifiedMemory
//      ////check if data as regular pointer (CPU memory)
//      delete this->_data[j];
//      // or on Unified Memory
//      // delete_memory(this->_data[j]);
//   }
//#else
   unsigned index = 0;
   for (unsigned i = 0; (i < this->_activeBlocks) && (index < this->_size); 
	i++) {
      for (unsigned j = 0; (j < blockSize) && (index < this->_size); j++) {
	 index++;
	 delete this->_blocksArray[i][j];
      }
   }
//#endif
}

#endif
#endif
