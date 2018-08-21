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


#include "VectorOstream.h"


std::ostream & operator<<(std::ostream &os, std::vector<int> &v)
{
   std::vector<int>::iterator end = v.end();
   std::vector<int>::iterator begin = v.begin();

   for (std::vector<int>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}

std::ostream & operator<<(std::ostream &os, const std::vector<int> &v)
{
   std::vector<int>::const_iterator end = v.end();
   std::vector<int>::const_iterator begin = v.begin();

   for (std::vector<int>::const_iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}

std::ostream & operator<<(std::ostream &os, std::vector<unsigned> &v)
{
   std::vector<unsigned>::iterator end = v.end();
   std::vector<unsigned>::iterator begin = v.begin();

   for (std::vector<unsigned>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}

std::ostream & operator<<(std::ostream &os, const std::vector<unsigned> &v)
{
   std::vector<unsigned>::const_iterator end = v.end();
   std::vector<unsigned>::const_iterator begin = v.begin();

   for (std::vector<unsigned>::const_iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}

std::ostream & operator<<(std::ostream &os, std::vector<short> &v)
{
   std::vector<short>::iterator end = v.end();
   std::vector<short>::iterator begin = v.begin();

   for (std::vector<short>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}


std::ostream & operator<<(std::ostream &os, const std::vector<short> &v)
{
   std::vector<short>::const_iterator end = v.end();
   std::vector<short>::const_iterator begin = v.begin();

   for (std::vector<short>::const_iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}


std::ostream & operator<<(std::ostream &os, std::vector<float> &v)
{
   std::vector<float>::iterator end = v.end();
   std::vector<float>::iterator begin = v.begin();

   for (std::vector<float>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}

std::ostream & operator<<(std::ostream &os, std::vector<double> &v)
{
   std::vector<double>::iterator end = v.end();
   std::vector<double>::iterator begin = v.begin();

   for (std::vector<double>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}


std::ostream & operator<<(std::ostream &os, const std::vector<float> &v)
{
   std::vector<float>::const_iterator end = v.end();
   std::vector<float>::const_iterator begin = v.begin();

   for (std::vector<float>::const_iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}


std::ostream & operator<<(std::ostream &os, std::vector<std::string> &v)
{
   std::vector<std::string>::iterator end = v.end();
   std::vector<std::string>::iterator begin = v.begin();

   for (std::vector<std::string>::iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}


std::ostream & operator<<(std::ostream &os, const std::vector<std::string> &v)
{
   std::vector<std::string>::const_iterator end = v.end();
   std::vector<std::string>::const_iterator begin = v.begin();

   for (std::vector<std::string>::const_iterator iter = begin; iter!=end; ++iter) {
      if(iter != begin) os<<", ";
      os<<(*iter);
   }
   return os;
}
