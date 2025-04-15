// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DuplicatePointerArray_H
#define DuplicatePointerArray_H
#include "Copyright.h"

#include "Array.h"

/* IMPORTANT
 * In GPU scenario: blockSize, blockIncrementSize do NOT play any role
 */
#if defined(HAVE_GPU) 
  #include "DuplicatePointerArray_GPU.h"
#endif

#ifdef USE_FLATARRAY_FOR_CONVENTIONAL_ARRAY
//Only C++11
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE,
          unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
using DuplicatePointerArray = DuplicatePointerArray_Flat<T, 0, blockIncrementSize>;
#else
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE, 
	  unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class DuplicatePointerArray : public Array<T*>
{
   public:
      DuplicatePointerArray();
      DuplicatePointerArray(const DuplicatePointerArray* rv);
      DuplicatePointerArray(const DuplicatePointerArray& rv);
      DuplicatePointerArray& operator=(const DuplicatePointerArray& rv);
      virtual void duplicate(std::unique_ptr<Array<T*> >& rv) const;
      virtual void duplicate(std::unique_ptr<DuplicatePointerArray<T, 
			     blockSize, blockIncrementSize> >& rv) const;
      virtual ~DuplicatePointerArray();

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
      void copyContents(const DuplicatePointerArray& rv);
};

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DuplicatePointerArray<T, blockSize, blockIncrementSize>::
DuplicatePointerArray()
   : Array<T*>(blockIncrementSize)
{
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DuplicatePointerArray<T, blockSize, blockIncrementSize>::
DuplicatePointerArray(const DuplicatePointerArray* rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array<T*>::copyContents(*rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DuplicatePointerArray<T, blockSize, blockIncrementSize>::
DuplicatePointerArray(const DuplicatePointerArray& rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array<T*>::copyContents(rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DuplicatePointerArray<T, blockSize, blockIncrementSize>& 
DuplicatePointerArray<T, blockSize, blockIncrementSize>::
operator=(const DuplicatePointerArray& rv)
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
void DuplicatePointerArray<T, blockSize, blockIncrementSize>::
duplicate(std::unique_ptr<Array<T*> >& rv) const
{
   rv.reset(new DuplicatePointerArray<T, blockSize, 
	    blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DuplicatePointerArray<T, blockSize, blockIncrementSize>::
duplicate(std::unique_ptr<DuplicatePointerArray<T, 
	  blockSize, blockIncrementSize> >& rv) const
{
   rv.reset(new DuplicatePointerArray<T, 
	    blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
DuplicatePointerArray<T, blockSize, blockIncrementSize>::
~DuplicatePointerArray()
{
   destructContents();
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DuplicatePointerArray<T, blockSize, blockIncrementSize>::
internalCopy(T*& lval, T*& rval)
{
   std::unique_ptr<T> dup;
   // Use decltype and SFINAE to detect the parameter type of duplicate
   using Args = decltype(&T::duplicate);
   if constexpr (std::is_same_v<Args, void (T::*)(std::unique_ptr<T>&) const>) {
      // lvalue reference version
      rval->duplicate(dup);
   } else {
      // rvalue reference version
      rval->duplicate(std::move(dup));
   }
   lval = dup.release();
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DuplicatePointerArray<T, blockSize, blockIncrementSize>::
destructContents()
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
