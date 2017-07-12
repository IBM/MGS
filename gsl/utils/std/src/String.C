// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "String.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
//#include <cstring>

// TUAN: should get rid of char*
//      we can provide String class wrapper to std::string class

// class-global constant intialization
unsigned int String::AllocIncr = 16;

// private function to shrink the size of an allocated string
void String::shrink() {
  char* temp;
  if ((_capacity - _size) > (AllocIncr * 1.5)) {  // Avoid the border condition
    _capacity = ((_size + AllocIncr - 1) / AllocIncr) * AllocIncr;
    temp = new char[_capacity];
    // memmove(temp, _data, _size);
    std::copy(_data, _data + _size, temp);
    delete[] _data;
    _data = temp;
  }
}

// constructor
String::String() {
  _size = 0;
  _capacity = AllocIncr;
  _data = new char[_capacity];
  _data[0] = '\x00';
}

String::String(const String& str) {
  _size = str._size;
  _capacity = str._capacity;
  _data = new char[_capacity];
  // memmove(_data, str._data, _size);
  std::copy(str._data, str._data + _size, _data);
}

String::String(const char* cstr) {
  int length = 0;
  while (cstr[length] != '\0')  //  remove ;
  {
    length++;
  }
  //_size = strlen(cstr);
  _size = length;
  _capacity = ((_size + AllocIncr - 1) / AllocIncr) * AllocIncr;
  _data = new char[_capacity];
  // memmove(_data, cstr, _size);
  std::copy(cstr, cstr + _size, _data);
}

String::String(char fillCh, unsigned int count) {
  _capacity = ((count + AllocIncr - 1) / AllocIncr) * AllocIncr;
  _size = _capacity;
  _data = new char[_capacity];
  // memset(_data, fillCh, count);
  std::fill(_data, _data + count, fillCh);
}

// destructor
String::~String() { delete[] _data; }

// value return methods
unsigned int String::size() { return _size; }

unsigned int String::capacity() { return _capacity; }

// Function to return a blank string
String empty() {
  static String emptyStr;
  return emptyStr;
}

// copy String to c-string method
void String::copy(char* cstr, unsigned int max) {
  unsigned int copyLen;

  if (max == 0) {
    return;
  }
  if (_size >= max) {
    copyLen = max - 1;
  } else {
    copyLen = _size;
  }
  // memmove(cstr, _data, copyLen);
  std::copy(_data, _data + copyLen, cstr);
  cstr[copyLen] = '\x00';
}

// create a c-string from String method
const char* String::c_str() {
  if (_size == _capacity) {
    char* temp;
    _capacity += AllocIncr;
    temp = new char[_capacity];

    // memmove(temp, _data, _size);
    std::copy(_data, _data + _size, temp);
    delete[] _data;
    _data = temp;
  }
  _data[_size] = '\x00';
  return _data;
}

// assignment method
String& String::operator=(const String& str) {
  _size = str._size;
  _capacity = str._capacity;
  delete[] _data;
  _data = new char[_capacity];
  // memmove(_data, str._data, _size);
  std::copy(str._data, str._data + _size, _data);
  return *this;
}

// concatenation methods
String operator+(const String& str1, const String& str2) {
  unsigned long totalLen;
  unsigned int newLen, newSiz, copyLen;
  String tempStr;
  char* temp;

  totalLen = str1._size + str2._size;

  tempStr = str1;
  copyLen = str2._size;

  newLen = tempStr._size + str2._size;
  newSiz = tempStr._capacity + str2._capacity;

  temp = new char[newSiz];

  // memmove(temp, tempStr._data, tempStr._size);
  std::copy(tempStr._data, tempStr._data + tempStr._size, temp);
  delete[] tempStr._data;
  tempStr._data = temp;

  // memmove(&tempStr._data[tempStr._size], str2._data, copyLen);
  std::copy(str2._data, str2._data + copyLen, &tempStr._data[tempStr._size]);

  tempStr._size = newLen;
  tempStr._capacity = newSiz;

  tempStr.shrink();

  return tempStr;
}

String& String::operator+=(const String& str) {
  unsigned long totalLen;
  unsigned int newLen, newSiz, copyLen;
  char* temp;

  totalLen = str._size + _size;

  copyLen = str._size;
  newLen = (unsigned int)totalLen;
  newSiz = _capacity + str._capacity;

  temp = new char[newSiz];

  // memmove(temp, _data, _size);
  std::copy(_data, _data + _size, temp);
  delete[] _data;
  _data = temp;

  // memmove(&_data[_size], str._data, copyLen);
  std::copy(str._data, str._data + copyLen, &_data[_size]);

  _size = newLen;
  _capacity = newSiz;

  shrink();
  return *this;
}

// comparison methods
int String::operator<(const String& str) const {
  return (compare(str) == SC_LESS);
}

int String::operator>(const String& str) const {
  return (compare(str) == SC_GREATER);
}

int String::operator<=(const String& str) const {
  return (compare(str) != SC_GREATER);
}

int String::operator>=(const String& str) const {
  return (compare(str) != SC_LESS);
}

int String::operator==(const String& str) const {
  return (compare(str) == SC_EQUAL);
}

int String::operator!=(const String& str) const {
  return (compare(str) != SC_EQUAL);
}

int String::compare(const String& str) const {
  unsigned int index, minIndex;
  char ch1, ch2;

  if (_size == 0) {
    if (str._size == 0) {
      return SC_EQUAL;
    } else {
      return SC_LESS;
    }
  } else {
    if (str._size == 0) {
      return SC_GREATER;
    }
  }

  minIndex = _size <= str._size ? _size : str._size;

  index = 0;

  do {
    ch1 = _data[index];
    ch2 = str._data[index];

    if (ch1 == ch2) {
      ++index;
    } else {
      if (_data[index] < str._data[index]) {
        return SC_LESS;
      } else {
        return SC_GREATER;
      }
    }
  } while (index < minIndex);

  if (_size < str._size) {
    return SC_LESS;
  } else if (_size > str._size) {
    return SC_GREATER;
  }
  // else _size == str._size
  return SC_EQUAL;
}

// substring search methods
int String::find(const String& str, unsigned int& pos) {
  char* tempStr1, *tempStr2;
  unsigned int lastPos, searchLen, tempPos;
  bool found;

  tempStr1 = new char[_size + 1];

  // memmove(tempStr1, _data, _size);
  std::copy(_data, _data + _size, tempStr1);
  tempStr1[_size] = '\x00';

  tempStr2 = new char[str._size + 1];

  // memmove(tempStr2, str._data, str._size);
  std::copy(str._data, str._data + str._size, tempStr2);
  tempStr2[str._size] = '\x00';

  pos = 0;
  tempPos = 0;
  found = false;

  searchLen = str._size;
  lastPos = _size - searchLen;

  while ((tempPos <= lastPos) && !found) {
    // if (0 == strncmp(&tempStr1[tempPos], tempStr2, searchLen)) {
    if (std::string(&tempStr1[tempPos]) == std::string(tempStr2)) {
      pos = tempPos;
      found = true;
    } else {
      ++tempPos;
    }
  }

  delete[] tempStr1;
  delete[] tempStr2;

  if (found) {
    return pos;
  } else {
    return 0;
  }
}

// substring deletion method
void String::del(unsigned int pos, unsigned int count) {
  unsigned int copyPos;

  if (pos > _size) {
    return;
  }

  copyPos = pos + count;

  if (copyPos >= _size) {
    _data[pos] = 0;
  } else {
    while (copyPos <= _size) {
      _data[pos] = _data[copyPos];
      ++pos;
      ++copyPos;
    }
  }
  _size -= count;

  shrink();
}

// substring insertion methods
void String::insert(unsigned int pos, char ch) {
  char* temp;

  if (pos > _size) {
    return;
  }

  if (_size == _capacity) {
    _capacity += AllocIncr;
    temp = new char[_capacity];

    // memmove(temp, _data, _size);
    std::copy(_data, _data + _size, temp);
    delete[] _data;
    _data = temp;
  }

  if (pos < _size) {
    for (unsigned int col = _size + 1; col > pos; --col) {
      _data[col] = _data[col - 1];
    }
  }

  _data[pos] = ch;

  ++_size;
}

void String::insert(unsigned int pos, const String& str) {
  unsigned int sLen = str._size;

  if (sLen > 0) {
    for (unsigned int i = 0; i < sLen; ++i) {
      insert(pos, str._data[i]);
      ++pos;
    }
  }
}

// append method
void String::append(char ch) { insert(size(), ch); }

// substring retrieval method
String String::subStr(unsigned int start, unsigned int count) {
  String tempStr;
  char* temp;

  if ((start < _size) && (count > 0)) {
    for (unsigned int pos = 0; pos < count; ++pos) {
      if (tempStr._size == tempStr._capacity) {
        tempStr._capacity += AllocIncr;
        temp = new char[tempStr._capacity];

        // memmove(temp, tempStr._data, _size);
        std::copy(tempStr._data, tempStr._data + _size, temp);
        delete[] tempStr._data;
        tempStr._data = temp;
      }

      tempStr._data[pos] = _data[start + pos];

      ++tempStr._size;
    }
  }

  return tempStr;
}

// character retrieval method
char String::operator[](unsigned int pos) {
  if (pos >= _size) {
    return '\x00';
  }

  return _data[pos];
}

// case-modification methods
String String::toUpper() {
  String tempStr = *this;

  for (unsigned int Pos = 0; Pos < _size; ++Pos) {
    tempStr._data[Pos] = toupper(tempStr._data[Pos]);
  }

  return tempStr;
}

String String::toLower() {
  String tempStr = *this;

  for (unsigned int pos = 0; pos < _size; ++pos) {
    tempStr._data[pos] = tolower(tempStr._data[pos]);
  }

  return tempStr;
}

// stream I/O methods
std::istream& operator>>(std::istream& input, String& str) {
  char buffer;

  str = empty();

  for (input.get(buffer);
       (!input.eof()) && (buffer != '\0') && (buffer != ' ') &&
           (buffer != '\n') && (buffer != '\t');
       input.get(buffer)) {
    str.append(buffer);
  }

  return input;
}

std::ostream& operator<<(std::ostream& output, const String& str) {
  unsigned int index;

  for (index = 0; index < str._size; ++index) output << str._data[index];

  return output;
}
