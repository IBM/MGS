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

#ifndef DuplicatePointerArray_H
#define DuplicatePointerArray_H
#include "Copyright.h"

#include "Array.h"

/* IMPORTANT
 * In GPU scenario: blockSize, blockIncrementSize do NOT play any role
 */
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
#if ! (defined(HAVE_GPU) && defined(__NVCC__))
      virtual unsigned getBlockSize() const {
	 return blockSize;
      }
      virtual unsigned getBlockIncrementSize() const {
	 return blockIncrementSize;
      }
#endif
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
   rval->duplicate(dup);
   lval = dup.release();
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void DuplicatePointerArray<T, blockSize, blockIncrementSize>::
destructContents()
{
#if defined(HAVE_GPU) && defined(__NVCC__)
   for (unsigned j = 0; j < this->_size; j++) {
      //TUAN TODO FIX
      //use this->_mem_location == MemLocation::CPU  or MemLocation::UnifiedMemory
      ////check if data as regular pointer (CPU memory)
      delete this->_data[j];
      // or on Unified Memory
      // delete_memory(this->_data[j]);
   }
#else
   unsigned index = 0;
   for (unsigned i = 0; (i < this->_activeBlocks) && (index < this->_size); 
	i++) {
      for (unsigned j = 0; (j < blockSize) && (index < this->_size); j++) {
	 index++;
	 delete this->_blocksArray[i][j];
      }
   }
#endif
}

#endif
