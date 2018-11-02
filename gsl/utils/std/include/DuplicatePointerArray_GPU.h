// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-10-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef DUPLICATEPOINTERARRAY_GPU_H
#define DUPLICATEPOINTERARRAY_GPU_H
template <class T, int memLocation = 0, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class DuplicatePointerArray_Flat : public Array_Flat<T*, memLocation>
{
   public:
      DuplicatePointerArray_Flat();
      DuplicatePointerArray_Flat(const DuplicatePointerArray_Flat* rv);
      DuplicatePointerArray_Flat(const DuplicatePointerArray_Flat& rv);
      DuplicatePointerArray_Flat& operator=(const DuplicatePointerArray_Flat& rv);
      virtual void duplicate(std::unique_ptr<Array_Flat<T*> >& rv) const;
      virtual void duplicate(std::unique_ptr<DuplicatePointerArray_Flat<T, 
			     memLocation, blockIncrementSize> >& rv) const;
      virtual ~DuplicatePointerArray_Flat();

      virtual void clear() {
	 destructContents();
	 Array_Flat<T*, memLocation>::clear();
      }

   protected:
      virtual void internalCopy(T*& lval, T*& rval);
      void destructContents();
      void copyContents(const DuplicatePointerArray_Flat& rv);
};

template <class T, int memLocation, unsigned blockIncrementSize>
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
DuplicatePointerArray_Flat()
   : Array_Flat<T*, memLocation>(blockIncrementSize)
{
}

template <class T, int memLocation, unsigned blockIncrementSize>
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
DuplicatePointerArray_Flat(const DuplicatePointerArray_Flat* rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array_Flat<T*, memLocation>::copyContents(*rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
DuplicatePointerArray_Flat(const DuplicatePointerArray_Flat& rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array_Flat<T*, memLocation>::copyContents(rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>& 
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
operator=(const DuplicatePointerArray_Flat& rv)
{
   if (this == &rv) {
      return *this;
   }
   // !!!! Important destruct before op=
   destructContents();
   Array_Flat<T*, memLocation>::operator=(rv);
   return *this;
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
duplicate(std::unique_ptr<Array_Flat<T*> >& rv) const
{
   rv.reset(new DuplicatePointerArray_Flat<T, memLocation, 
	    blockIncrementSize>(this));
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
duplicate(std::unique_ptr<DuplicatePointerArray_Flat<T, 
	  memLocation, blockIncrementSize> >& rv) const
{
   rv.reset(new DuplicatePointerArray_Flat<T, 
	    memLocation, blockIncrementSize>(this));
}

template <class T, int memLocation, unsigned blockIncrementSize>
DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
~DuplicatePointerArray_Flat()
{
   destructContents();
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
internalCopy(T*& lval, T*& rval)
{
   std::unique_ptr<T> dup;
   rval->duplicate(dup);
   lval = dup.release();
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DuplicatePointerArray_Flat<T, memLocation, blockIncrementSize>::
destructContents()
{
   for (unsigned j = 0; j < this->_size; j++) {
      //TUAN TODO FIX
      //use this->_mem_location == MemLocation::CPU  or MemLocation::UnifiedMemory
      ////check if data as regular pointer (CPU memory)
      delete this->_data[j];
      // or on Unified Memory
      // delete_memory(this->_data[j]);
   }
}
#endif
