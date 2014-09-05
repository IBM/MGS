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

#ifndef PARSER_H_
#define PARSER_H_
#include "Copyright.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
class Parser {

 public:
  class Option {
  public:
    static char SHORT_NAME_NONE;
    static std::string LONG_NAME_NONE;
    static Option OPTION_NONE;
    enum Type {
      TYPE_NONE, TYPE_OPTIONAL, TYPE_REQUIRED
    };
  private:
    char fieldShortName;
    std::string fieldLongName;
    Type fieldType;
  public:
    Option(char shortName, std::string longName, Type type);
    Option(const Option &option); 
    virtual ~Option();
    //
    char getShortName() const;
    void setShortName(char shortName);
    std::string getLongName() const;
    void setLongName(const std::string &longName);
    Type getType() const;
    void setType(Type type);
    //
    bool operator==(const Option &option);
    Option& operator=(const Option &option);
  };

  class Parameter {
  private:
    Option fieldOption;
    std::string fieldValue;
  public:
    Parameter(const Option &option, const std::string &value);
    Parameter(const Parameter &parameter);
    Parameter(const Option &option);
    virtual ~Parameter();
    //
    Option const & getOption() const;
    void setOption(const Option &option);
    std::string getValue() const;
    void setValue(const std::string &value);
    //
    Parameter &operator=(const Parameter &parameter);
  };	

  typedef std::string String;
  typedef std::vector<String> StringVector;
  typedef std::vector<Option> OptionVector;
  typedef std::vector<Parameter> ParameterVector;
  class Exception {
  private:
    String fieldMessage;
  public:
    Exception(String message) : fieldMessage(message) {}
    ~Exception() {}
    String getMessage() {
      return(fieldMessage);
    }
  }; 
 private:
  OptionVector fieldOptions;
 public:
  Parser();
  virtual ~Parser();
  //
  int getOptionCount() const;
  Option  const & getOption(int i) const;
  void addOption(Option option);
  void removeOption(Option option);
  void help();
  //
  ParameterVector parse(int argc, char *argv[]);
 private:
  OptionVector &getOptions();
  int countOptions(String arg);
  int findOption(String arg);
  int countOptions(char c);
  int findOption(char c);
  int countType(String arg, Option::Type type);
  int countArgs(StringVector &args, StringVector::size_type start);	
};

inline Parser::Parser() {}
inline Parser::~Parser() {}
inline int Parser::getOptionCount() const {
	return(fieldOptions.size());
}
inline Parser::Option const & Parser::getOption(int i) const {
	return(fieldOptions.at(i));
}
inline void Parser::addOption(Option option) {
	// Doesn't throw an exception if same option is added twice...
	fieldOptions.push_back(option);
}
inline void Parser::removeOption(Option option) {
	// Doesn't throw an exception if non-existent option is removed...
        fieldOptions.erase(find(fieldOptions.begin(), fieldOptions.end(), option));
}

inline Parser::OptionVector &Parser::getOptions() {
	return(fieldOptions);
}

		
inline char Parser::Option::getShortName() const {
  return(fieldShortName);
}
inline void Parser::Option::setShortName(char shortName) {
  fieldShortName = shortName;
}
inline std::string Parser::Option::getLongName() const {
  return(fieldLongName);
}
inline void Parser::Option::setLongName(const std::string &longName) {
  fieldLongName = longName;
}
inline Parser::Option::Type Parser::Option::getType() const {
  return(fieldType);
}
inline void Parser::Option::setType(Type type) {
  fieldType = type;
}

inline bool Parser::Option::operator==(const Parser::Option &option) {
  return(getShortName() == option.getShortName() &&
	 getLongName() == option.getLongName() &&
	 getType() == option.getType());
}

inline Parser::Parameter::Parameter(const Parser::Option &option, const std::string &value) :
  fieldOption(option), fieldValue(value) {
}
inline Parser::Parameter::Parameter(const Parameter &parameter) :
  fieldOption(parameter.getOption()), fieldValue(parameter.getValue()) {
}
inline Parser::Parameter::Parameter(const Parser::Option &option) :
  fieldOption(option), fieldValue("") {
}
inline Parser::Parameter::~Parameter() {}
inline Parser::Option const & Parser::Parameter::getOption() const {
  return(fieldOption);
}
inline void Parser::Parameter::setOption(const Parser::Option &option) {
  fieldOption = option;
}
inline std::string Parser::Parameter::getValue() const {
  return(fieldValue);
}
inline void Parser::Parameter::setValue(const std::string &value) {
  fieldValue = value;
}
inline Parser::Parameter &Parser::Parameter::operator=(const Parser::Parameter &parameter) {
  setOption(parameter.getOption());
  setValue(parameter.getValue());
  return(*this);
}

#endif /*PARSER_H_*/
