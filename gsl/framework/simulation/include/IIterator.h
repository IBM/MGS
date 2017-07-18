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

#ifndef IIterator_H
#define IIterator_H
#include "Copyright.h"

#include "ShallowArray.h"

template <class T>
class IIterator
{
public:
    void push_back(T*);
    T* getFirst();
    T* getNext();
    ~IIterator();
    IIterator();
    IIterator(int);
    void sort(T);
private:
    typename ShallowArray <T*>::iterator _iter;
    ShallowArray <T*> _classArray;
};

template <class T> 
IIterator<T>::IIterator(int num)
{
   for (int i = 0; i < num; i++) {
      _classArray.push_back(new T);
   }
}

template <class T> 
IIterator<T>::~IIterator()
{
   for (_iter = _classArray.begin(); _iter != _classArray.end(); _iter++)
      delete (*_iter);
//   delete [] _classArray;
}

template <class T> 
void IIterator<T>::push_back(T* cp)
{
   _classArray.push_back(cp);
}

template <class T> 
T* IIterator<T>::getFirst()
{
   if (_classArray.size() == 0)
      return NULL;
   else {
      _iter = _classArray.begin();
      return *_iter;
   }
}

template <class T> 
T* IIterator<T>::getNext()
{
   if (++_iter != _classArray.end())
      return *_iter;
   else
      return NULL;
}
#endif
