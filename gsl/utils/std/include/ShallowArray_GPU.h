#ifndef SHALLOWARRAY_GPU_H
#define SHALLOWARRAY_GPU_H

template <class T, int memLocation=0, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class ShallowArray_Flat : public Array_Flat<T, memLocation>
{
  public:
  ShallowArray_Flat();
  ShallowArray_Flat(const ShallowArray_Flat* rv);
  ShallowArray_Flat(const ShallowArray_Flat& rv);
  ShallowArray_Flat& operator=(const ShallowArray_Flat& rv);
  //virtual void duplicate(
  //    std::unique_ptr<Array_Flat<T> >& rv) const;
  virtual void duplicate(
      std::unique_ptr<Array_Flat<T, memLocation> >& rv) const;
  //virtual void duplicate(
  //    std::unique_ptr<ShallowArray_Flat<T, memLocation> >& rv) const;
  virtual void duplicate(
      std::unique_ptr<ShallowArray_Flat<T, memLocation, blockIncrementSize> >& rv) const;
  virtual ~ShallowArray_Flat();

  protected:
  virtual void internalCopy(T& lval, T& rval);
  void destructContents();
  void copyContents(const ShallowArray_Flat& rv);
};

template <class T, int memLocation, unsigned blockIncrementSize>
ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat()
    : Array_Flat<T, memLocation>(blockIncrementSize)
{
}

template <class T, int memLocation, unsigned blockIncrementSize>
ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat(
    const ShallowArray_Flat* rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array_Flat<T, memLocation>::copyContents(*rv);
  copyContents(*rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat(
    const ShallowArray_Flat& rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array_Flat<T, memLocation>::copyContents(rv);
  copyContents(rv);
}

template <class T, int memLocation, unsigned blockIncrementSize>
ShallowArray_Flat<T, memLocation, blockIncrementSize>&
    ShallowArray_Flat<T, memLocation, blockIncrementSize>::
        operator=(const ShallowArray_Flat& rv)
{
  if (this == &rv)
  {
    return *this;
  }
  Array_Flat<T, memLocation>::operator=(rv);
  destructContents();
  copyContents(rv);
  return *this;
}

//template <class T, int memLocation, unsigned blockIncrementSize>
//void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
//    std::unique_ptr<Array_Flat<T> >& rv) const
//{
//  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
//}

template <class T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
    std::unique_ptr<Array_Flat<T, memLocation> >& rv) const
{
  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
}

//template <class T, int memLocation, unsigned blockIncrementSize>
//void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
//    std::unique_ptr<ShallowArray_Flat<T, memLocation> >& rv) const
//{
//  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
//}

template <class T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
    std::unique_ptr<ShallowArray_Flat<T, memLocation, blockIncrementSize> >& rv) const
{
  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
}

template <class T, int memLocation, unsigned blockIncrementSize>
ShallowArray_Flat<T, memLocation, blockIncrementSize>::~ShallowArray_Flat()
{
  destructContents();
}

template <class T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::internalCopy(T& lval,
                                                                  T& rval)
{
  lval = rval;
}

template <class T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::copyContents(
    const ShallowArray_Flat& rv)
{
}

template <class T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::destructContents()
{
}

#endif
