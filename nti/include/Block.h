// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
/*
 * Block.h
 *
 *  Created on: Feb 13, 2010
 *      Author: wagnerjo
 */

#ifndef BLOCK_H_
#define BLOCK_H_

#include <mpi.h>

template<typename T> class Block {
 private:
  int fieldSize;
  int fieldCount;
  T *fieldValues;
 public:
  Block(int size = 100) {
    fieldSize = size;
    fieldCount = 0;
    fieldValues = new T[getSize()];
  }
  Block(const Block &block) {
    fieldSize = block.getSize();
    fieldCount = block.getCount();
    fieldValues = new T[getSize()];
    for (int i = 0; i < getCount(); i++) {
      fieldValues[i] = block.getValue(i);
    }
  }
  virtual ~Block() {
    delete[] fieldValues;
  }
  //
  int getSize() const {
    return(fieldSize);
  }
  int getCount() const {
    return(fieldCount);
  }
  void setCount(int count) {
    while (getCount() < count) push_back(T());
    fieldCount = count;
  }
  // For MPI writes...
  T *getData() {
    return(fieldValues);
  }
  T &getValue(int i) {
    assert(0 <= i && i < getCount());
    return(fieldValues[i]);
  }
  const T &getValue(int i) const {
    assert(0 <= i && i < getCount());
    return(fieldValues[i]);
  }
  void setValue(int i, const T &t) {
    assert(0 <= i && i < getCount());
    fieldValues[i] = t;
  }
  int push_back(const T &t) {
    assert(getCount() < getSize());
    fieldValues[fieldCount++] = t;
    return(getCount() - 1);
  }
  T &operator[](int i) {
    assert(0 <= i && i < getCount());
    return(fieldValues[i]);
  }
  const T &operator[](int i) const {
    assert(0 <= i && i < getCount());
    return(fieldValues[i]);
  }
  //
  class iterator {
  private:
    T *fieldValue;
  public:
    iterator(T *value) {
      fieldValue = value;
    }
    iterator(const iterator &iterator) {
      fieldValue = iterator.getValue();
    }
    iterator &operator=(const iterator &iterator) {
      return(fieldValue = iterator.getValue());
    }
    T *getValue() const {
      return(fieldValue);
    }
    bool operator==(const iterator &iterator) const {
      return(getValue() == iterator.getValue());
    }
    bool operator!=(const iterator &iterator) const {
      return(getValue() != iterator.getValue());
    }
    T &operator*() {
      return(*fieldValue);
    }
    const T &operator*() const {
      return(*fieldValue);
    }
    T *operator->() {
      return(fieldValue);
    }
    const T *operator->() const {
      return(fieldValue);
    }
    typename Block<T>::iterator &operator++() {
      fieldValue++;
      return(*this);
    }
    const typename Block<T>::iterator &operator++() const {
      fieldValue++;
      return(*this);
    }
    typename Block<T>::iterator operator++(int) {
      typename Block<T>::iterator iterator(fieldValue);
      fieldValue++;
      return(iterator);
    }
    const typename Block<T>::iterator operator++(int) const {
      typename Block<T>::iterator iterator(fieldValue);
      fieldValue++;
      return(iterator);
    }
  };
  iterator begin() const {
    return(iterator(&fieldValues[0]));
  }
  iterator end() const {
    return(iterator(&fieldValues[getCount()]));
  }
 private:
  Block<T> &operator=(const Block<T> &);
};

#endif /* BLOCK_H_ */
