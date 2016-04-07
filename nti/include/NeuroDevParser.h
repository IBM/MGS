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
// ================================================================

#ifndef NEURODEVPARSER_H_
#define NEURODEVPARSER_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
class NeuroDevParser
{

  public:
  class Option
  {
public:
    static char ND_SHORT_NAME_NONE;
    static std::string ND_LONG_NAME_NONE;
    static Option ND_OPTION_NONE;
    enum Type
    {
      TYPE_NONE,
      TYPE_OPTIONAL,
      TYPE_REQUIRED
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
    Option &operator=(const Option &option);
  };

  class Parameter
  {
private:
    Option fieldOption;
    std::string fieldValue;

public:
    Parameter(const Option &option, const std::string &value);
    Parameter(const Parameter &parameter);
    Parameter(const Option &option);
    virtual ~Parameter();
    //
    Option const &getOption() const;
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
  class Exception
  {
private:
    String fieldMessage;

public:
    Exception(String message) : fieldMessage(message) {}
    ~Exception() {}
    String getMessage() { return (fieldMessage); }
  };

  private:
  OptionVector fieldOptions;

  public:
  NeuroDevParser();
  virtual ~NeuroDevParser();
  //
  int getOptionCount() const;
  Option const &getOption(int i) const;
  void addOption(Option option);
  void removeOption(Option option);
  void help();
  //
  ParameterVector parse(int argc, char *argv[]);
  ParameterVector parse(std::vector<std::string> &tokens);

  private:
  OptionVector &getOptions();
  int countOptions(String arg);
  int findOption(String arg);
  int countOptions(char c);
  int findOption(char c);
  int countType(String arg, Option::Type type);
  int countArgs(StringVector &args, StringVector::size_type start);
};

inline NeuroDevParser::NeuroDevParser() {}
inline NeuroDevParser::~NeuroDevParser() {}
inline int NeuroDevParser::getOptionCount() const
{
  return (fieldOptions.size());
}
inline NeuroDevParser::Option const &NeuroDevParser::getOption(int i) const
{
  return (fieldOptions.at(i));
}
inline void NeuroDevParser::addOption(Option option)
{
  // Doesn't throw an exception if same option is added twice...
  fieldOptions.push_back(option);
}
inline void NeuroDevParser::removeOption(Option option)
{
  // Doesn't throw an exception if non-existent option is removed...
  fieldOptions.erase(find(fieldOptions.begin(), fieldOptions.end(), option));
}

inline NeuroDevParser::OptionVector &NeuroDevParser::getOptions()
{
  return (fieldOptions);
}

inline char NeuroDevParser::Option::getShortName() const
{
  return (fieldShortName);
}
inline void NeuroDevParser::Option::setShortName(char shortName)
{
  fieldShortName = shortName;
}
inline std::string NeuroDevParser::Option::getLongName() const
{
  return (fieldLongName);
}
inline void NeuroDevParser::Option::setLongName(const std::string &longName)
{
  fieldLongName = longName;
}
inline NeuroDevParser::Option::Type NeuroDevParser::Option::getType() const
{
  return (fieldType);
}
inline void NeuroDevParser::Option::setType(Type type) { fieldType = type; }

inline bool NeuroDevParser::Option::operator==(
    const NeuroDevParser::Option &option)
{
  return (getShortName() == option.getShortName() &&
          getLongName() == option.getLongName() &&
          getType() == option.getType());
}

inline NeuroDevParser::Parameter::Parameter(
    const NeuroDevParser::Option &option, const std::string &value)
    : fieldOption(option), fieldValue(value)
{
}
inline NeuroDevParser::Parameter::Parameter(const Parameter &parameter)
    : fieldOption(parameter.getOption()), fieldValue(parameter.getValue())
{
}
inline NeuroDevParser::Parameter::Parameter(
    const NeuroDevParser::Option &option)
    : fieldOption(option), fieldValue("")
{
}
inline NeuroDevParser::Parameter::~Parameter() {}
inline NeuroDevParser::Option const &NeuroDevParser::Parameter::getOption()
    const
{
  return (fieldOption);
}
inline void NeuroDevParser::Parameter::setOption(
    const NeuroDevParser::Option &option)
{
  fieldOption = option;
}
inline std::string NeuroDevParser::Parameter::getValue() const
{
  return (fieldValue);
}
inline void NeuroDevParser::Parameter::setValue(const std::string &value)
{
  fieldValue = value;
}
inline NeuroDevParser::Parameter &NeuroDevParser::Parameter::operator=(
    const NeuroDevParser::Parameter &parameter)
{
  setOption(parameter.getOption());
  setValue(parameter.getValue());
  return (*this);
}

#endif /*NEURODEVPARSER_H_*/
