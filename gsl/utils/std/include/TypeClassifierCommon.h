// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef TYPECLASSIFIERCOMMON_H
#define TYPECLASSIFIERCOMMON_H
#include "Copyright.h"

#include <string>
#include <vector>

class ConstantType;
class NodeType;
class EdgeType;
class FunctorType;
class TriggerType;
class StructType;
class VariableType;
class NodeSet;

template <>
const std::string TypeClassifier<ConstantType>::getName();

template <>
const std::string TypeClassifier<NodeType>::getName();

template <>
const std::string TypeClassifier<EdgeType>::getName();

template <>
const std::string TypeClassifier<FunctorType>::getName();

template <>
const std::string TypeClassifier<TriggerType>::getName();

template <>
const std::string TypeClassifier<StructType>::getName();

template <>
const std::string TypeClassifier<VariableType>::getName();

template <>
const std::string TypeClassifier<bool>::getName();

template <>
const std::string TypeClassifier<std::vector<bool> >::getName();

template <>
const std::string TypeClassifier<double>::getName();

template <>
const std::string TypeClassifier<std::vector<double> >::getName();

template <>
const std::string TypeClassifier<float>::getName();

template <>
const std::string TypeClassifier<std::vector<float> >::getName();

template <>
const std::string TypeClassifier<int>::getName();

template <>
const std::string TypeClassifier<std::vector<int> >::getName();

template <>
const std::string TypeClassifier<NodeSet>::getName();

template <>
const std::string TypeClassifier<std::vector<NodeSet*> >::getName();

template <>
const std::string TypeClassifier<std::string>::getName();

template <>
const std::string TypeClassifier<unsigned>::getName();

template <>
const std::string TypeClassifier<std::vector<unsigned> >::getName();


template <>
const std::string TypeClassifier<ConstantType>::getName() {
   return "ConstantType";
};

template <>
const std::string TypeClassifier<NodeType>::getName() {
   return "NodeType";
};

template <>
const std::string TypeClassifier<EdgeType>::getName() {
   return "EdgeType";
};

template <>
const std::string TypeClassifier<FunctorType>::getName() {
   return "FunctorType";
};

template <>
const std::string TypeClassifier<TriggerType>::getName() {
   return "TriggerType";
};

template <>
const std::string TypeClassifier<StructType>::getName() {
   return "StructType";
};

template <>
const std::string TypeClassifier<VariableType>::getName() {
   return "VariableType";
};

template <>
const std::string TypeClassifier<bool>::getName() {
   return "bool";
};

template <>
const std::string TypeClassifier<std::vector<bool> >::getName() {
   return "vector of bool";
};

template <>
const std::string TypeClassifier<double>::getName() {
   return "double";
};

template <>
const std::string TypeClassifier<std::vector<double> >::getName() {
   return "vector of double";
};

template <>
const std::string TypeClassifier<float>::getName() {
   return "float";
};

template <>
const std::string TypeClassifier<std::vector<float> >::getName() {
   return "vector of float";
};

template <>
const std::string TypeClassifier<int>::getName() {
   return "int";
};

template <>
const std::string TypeClassifier<std::vector<int> >::getName() {
   return "vector of int";
};

template <>
const std::string TypeClassifier<NodeSet>::getName() {
   return "NodeSet";
};

template <>
const std::string TypeClassifier<std::vector<NodeSet*> >::getName() {
   return "vector of NodeSet*";
};

template <>
const std::string TypeClassifier<std::string>::getName() {
   return "string";
};

template <>
const std::string TypeClassifier<unsigned>::getName() {
   return "unsigned";
};

template <>
const std::string TypeClassifier<std::vector<unsigned> >::getName() {
   return "vector of unsigned";
};

#endif
