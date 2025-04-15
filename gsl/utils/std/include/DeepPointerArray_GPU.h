// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DEEPPOINTERARRAY_GPU_H
#define DEEPPOINTERARRAY_GPU_H

template <class T, int memLocation = 0, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class DeepPointerArray_Flat : public Array_Flat<T*, memLocation>
{
   public:
      DeepPointerArray_Flat();
      DeepPointerArray_Flat(const DeepPointerArray_Flat* rv);
      DeepPointerArray_Flat(const DeepPointerArray_Flat& rv);
      DeepPointerArray_Flat& operator=(const DeepPointerArray_Flat& rv);
      virtual void duplicate(std::unique_ptr<Array_Flat<T*> >& rv) const;
      virtual void duplicate(std::unique_ptr<DeepPointerArray_Flat<T, 
			     memLocation, blockIncrementSize> >& rv) const;
      virtual ~DeepPointerArray_Flat();

      virtual void clear() {
	 destructContents();
	 Array_Flat<T*, memLocation>::clear();
      }

   protected:
      virtual void internalCopy(T*& lval, T*& rval);
      void destructContents();
      void copyContents(const DeepPointerArray_Flat& rv);
};

template <class T, int memLocation, unsigned blockIncrementSize>
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::DeepPointerArray_Flat()
   : Array_Flat<T*, memLocation>(blockIncrementSize)
{
}

template <class T, int memLocation, unsigned blockIncrementSize>
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::DeepPointerArray_Flat(
   const DeepPointerArray_Flat* rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array_Flat<T*, memLocation>::copyContents(*rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::DeepPointerArray_Flat(
   const DeepPointerArray_Flat& rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in copyContents
{
   Array_Flat<T*, memLocation>::copyContents(rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>& 
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::operator=(
   const DeepPointerArray_Flat& rv)
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
void DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
   std::unique_ptr<Array_Flat<T*> >& rv) const
{
   rv.reset(new DeepPointerArray_Flat<T, memLocation, blockIncrementSize>(this));
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
   std::unique_ptr<
   DeepPointerArray_Flat<T, memLocation, blockIncrementSize> >& rv) const
{
   rv.reset(new DeepPointerArray_Flat<T, memLocation, blockIncrementSize>(this));
}

template <class T, int memLocation, unsigned blockIncrementSize>
DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::~DeepPointerArray_Flat()
{
   destructContents();
}

template <class T, int memLocation, unsigned blockIncrementSize>
void DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::internalCopy(
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

template <class T, int memLocation, unsigned blockIncrementSize>
void DeepPointerArray_Flat<T, memLocation, blockIncrementSize>::destructContents()
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
