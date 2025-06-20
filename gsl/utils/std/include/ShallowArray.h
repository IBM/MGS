// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ShallowArray_H
#define ShallowArray_H
#include "Copyright.h"

#include "Array.h"
#include "ArrayException.h"

/* IMPORTANT
 * In GPU scenario: blockSize, blockIncrementSize do NOT play any role
 */
#if defined(HAVE_GPU) 
  #include "ShallowArray_GPU.h"
#endif

#ifdef USE_FLATARRAY_FOR_CONVENTIONAL_ARRAY
//Only C++11
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE,
          unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
using ShallowArray = ShallowArray_Flat<T, 0, blockIncrementSize>;
#else
template <class T, unsigned blockSize = SUGGESTEDARRAYBLOCKSIZE,
          unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class ShallowArray : public Array<T>
{
  public:
  ShallowArray();
  ShallowArray(const ShallowArray* rv);
  ShallowArray(const ShallowArray& rv);
  ShallowArray& operator=(const ShallowArray& rv);
  virtual void duplicate(std::unique_ptr<Array<T> >& rv) const;
  virtual void duplicate(
      std::unique_ptr<ShallowArray<T, blockSize, blockIncrementSize> >& rv) const;
  virtual ~ShallowArray();

  protected:
  virtual void internalCopy(T& lval, T& rval);
//#if ! (defined(HAVE_GPU) && defined(__NVCC__))
  virtual unsigned getBlockSize() const { return blockSize; }
  virtual unsigned getBlockIncrementSize() const { return blockIncrementSize; }
//#endif
  void destructContents();
  void copyContents(const ShallowArray& rv);
};

template <class T, unsigned blockSize, unsigned blockIncrementSize>
ShallowArray<T, blockSize, blockIncrementSize>::ShallowArray()
    : Array<T>(blockIncrementSize)
{
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
ShallowArray<T, blockSize, blockIncrementSize>::ShallowArray(
    const ShallowArray* rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array<T>::copyContents(*rv);
  copyContents(*rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
ShallowArray<T, blockSize, blockIncrementSize>::ShallowArray(
    const ShallowArray& rv)
//   : Array<T>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array<T>::copyContents(rv);
  copyContents(rv);
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
ShallowArray<T, blockSize, blockIncrementSize>&
    ShallowArray<T, blockSize, blockIncrementSize>::
        operator=(const ShallowArray& rv)
{
  if (this == &rv)
  {
    return *this;
  }
  Array<T>::operator=(rv);
  destructContents();
  copyContents(rv);
  return *this;
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void ShallowArray<T, blockSize, blockIncrementSize>::duplicate(
    std::unique_ptr<Array<T> >& rv) const
{
  rv.reset(new ShallowArray<T, blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void ShallowArray<T, blockSize, blockIncrementSize>::duplicate(
    std::unique_ptr<ShallowArray<T, blockSize, blockIncrementSize> >& rv) const
{
  rv.reset(new ShallowArray<T, blockSize, blockIncrementSize>(this));
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
ShallowArray<T, blockSize, blockIncrementSize>::~ShallowArray()
{
  destructContents();
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void ShallowArray<T, blockSize, blockIncrementSize>::internalCopy(T& lval,
                                                                  T& rval)
{
  lval = rval;
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void ShallowArray<T, blockSize, blockIncrementSize>::copyContents(
    const ShallowArray& rv)
{
}

template <class T, unsigned blockSize, unsigned blockIncrementSize>
void ShallowArray<T, blockSize, blockIncrementSize>::destructContents()
{
}
#endif

#endif
