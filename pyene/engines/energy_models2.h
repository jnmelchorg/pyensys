#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <typeinfo>
#include <iterator>
#include <cmath>
#include <boost/tuple/tuple.hpp>
#include "boost/graph/adjacency_list.hpp"
#include "boost/variant.hpp"
#include "boost/functional/hash.hpp"
#include "ClpSimplex.hpp"
#include "CoinHelperFunctions.hpp"
#include "CoinTime.hpp"
#include <chrono>
#include <future>
#include <list>
#include <boost/unordered_map.hpp>

typedef boost::variant<bool, int, double, std::string> value_T;

class characteristic
{
    public:
        characteristic()
        {
            value = value_T(std::string("N/A"));
            name = "N/A";
        }
        characteristic(const std::string& na, const value_T& val, const bool& is_vector)
        {
            name = na;
            if (!is_vector) value=val;
            else
            {
                values.push_back(val);
                value = value_T(std::string("N/A"));
            }
        }
        
        void push_back(const value_T& val){values.push_back(val);}
        void insert(const std::vector<value_T>& vec){values.insert(values.end(), vec.begin(), vec.end());}
        void set_name(const std::string& na) {name = na;}
        void set_value(const value_T& val){value = val;}
        void clear_values(){values.clear();}
        
        std::string get_name() const {return name;}
        value_T get_value () const {return value;}
        std::vector<value_T> get_values() const {return values;}

    private:
        std::string name;
        value_T value;
        std::vector<value_T> values;
};

class information
{
    public:
        void set_characteristic(const characteristic& cha, const bool& is_vector) 
        {
            for (characteristic& element : characteristics)
            {
                if (element.get_name() == cha.get_name() && !is_vector)
                {
                    std::cout << "WARNING! Characteristic with name *" << cha.get_name() <<"* already exist with value *" << element.get_value() <<"*. The value will be replaced with *" << cha.get_value() <<"*." << std::endl;
                    element = cha;
                    return;
                }
                else if (element.get_name() == cha.get_name() && is_vector)
                {
                    element.insert(cha.get_values());
                    return;
                }
            }
            characteristics.push_back(cha);
        }
        void set_characteristics(const std::vector<characteristic>& cha){characteristics = cha;}
        void update_characteristic(const characteristic& cha)
        {
            for (characteristic& element : characteristics)
            {
                if (element.get_name() == cha.get_name())
                {
                    element = cha;
                    return;
                }
            }
            std::cout << "WARNING! Characteristic with name *" << cha.get_name() <<"* does not exist." << std::endl;
        }
        void set_value(const value_T& val){value = val;}

        characteristic get_characteristic(const std::string& na) const
        {
            for (const characteristic& cha: characteristics)
                if (cha.get_name() == na) return cha;
            std::cout << "Characteristic with name *" << na << "* does not exist" << std::endl;
            return characteristic("N/A", std::string("N/A"), false);
        }
        bool exist(const std::string& na) const
        {
            for (const characteristic& cha: characteristics)
                if (cha.get_name() == na) return true;
            return false;
        }
        std::vector<characteristic> get_characteristics() const {return characteristics;}
        value_T get_value() const {return value;}
        
        void clear_characteristics(){characteristics.clear();}
        void erase_characteristic(const std::string& na)
        {
            int pos = -1;
            bool exist = false;
            for (const characteristic& cha: characteristics)
            {
                pos++;
                if (cha.get_name() == na)
                {
                    exist = true;
                    break;
                }
            }
            if (exist) characteristics.erase(characteristics.begin()+pos);
        }
    
    private:
        std::vector<characteristic> characteristics;
        value_T value;
};

// custom specialization of std::hash in namespace std
namespace boost
{
    template<> struct hash<information>
    {
        std::size_t operator()(information const& info) const noexcept
        {
            std::list<std::size_t> h;
            for (const characteristic& cha : info.get_characteristics())
            {
                h.push_back(boost::hash_value(cha.get_name()));
                h.push_back(boost::hash_value(cha.get_value()));
                for (const value_T& val : cha.get_values())
                    h.push_back(boost::hash_value(val));
            }
            std::size_t h1 = 0;
            for (std::list<std::size_t>::iterator it = h.begin(); it != h.end(); ++it)
            {
                boost::hash_combine(h1, *it);
            }
            return h1;
        }
    };
}

namespace std
{
    template<> struct equal_to<information>
    {
        bool operator()(information const& lhs, information const& rhs) const
        {
            std::vector<characteristic> characteristics1 = lhs.get_characteristics();
            std::vector<characteristic> characteristics2 = rhs.get_characteristics();
            std::vector<characteristic>::iterator first1 = characteristics1.begin();
            std::vector<characteristic>::iterator last1 = characteristics1.end();
            std::vector<characteristic>::iterator first2 = characteristics2.begin();
            for (; first1 != last1; ++first1, ++first2) {
                if (!(first1->get_name() == first2->get_name() && first1->get_value() == first2->get_value() && first1->get_values() == first2->get_values())) {
                    return false;
                }
            }
            return true;
        }
    };
}

class parameters
{
    public:
        information get_parameter(const std::vector<characteristic>& characteristics, const std::vector<information>& parameters) const
        {
            for (const information& param : parameters)
            {
                if (param.get_characteristics().size() == characteristics.size() && is_subset(characteristics, param.get_characteristics()))
                    return param;
            }
            std::cout << "WARNING! information with the following characteristics not found: " << std::endl;
            for (const characteristic& cha : characteristics)
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }

            return information();
        }

        information get_parameter_type(const std::vector<characteristic>& characteristics, const std::string& type_var)
        {
            if (type_var == "network")
                return get_parameter(characteristics, network_parameters);
            else if (type_var == "tree")
                return get_parameter(characteristics, tree_parameters);
            else if (type_var == "model")
                return get_parameter(characteristics, model_parameters);
            else if (type_var == "connections")
                return get_parameter(characteristics, connections);
            else if (type_var == "outputs")
                return get_parameter(characteristics, outputs);
        }

        std::vector<information> get_multi_parameter(const std::vector<characteristic>& characteristics, const std::vector<information>& parameters, const bool silent) const
        {
            std::vector<information> m_info;
            for (const information& param : parameters)
            {
                if (is_subset(characteristics, param.get_characteristics()))
                    m_info.push_back(param);
            }
            if (m_info.size() > 0) return m_info;
            if (!silent)
            {
                std::cout << "WARNING! information with the following characteristics not found: " << std::endl;
                for (const characteristic& cha : characteristics)
                {
                    std::cout << cha.get_name() << " : ";
                    if (cha.get_values().size() > 0)
                    {
                        std::cout << "[.";
                        for (value_T& val : cha.get_values()) 
                            std::cout << val << ".";
                        std::cout << "]" << std::endl;
                    }
                    else std::cout << cha.get_value() << std::endl;
                }
            }
            return std::vector<information>();
        }

        std::vector<information> get_multi_parameter_type(const std::vector<characteristic>& characteristics, const std::string& type_var, const bool silent)
        {
            if (type_var == "network")
                return get_multi_parameter(characteristics, network_parameters, silent);
            else if (type_var == "tree")
                return get_multi_parameter(characteristics, tree_parameters, silent);
            else if (type_var == "model")
                return get_multi_parameter(characteristics, model_parameters, silent);
            else if (type_var == "connections")
                return get_multi_parameter(characteristics, connections, silent);
            else if (type_var == "outputs")
                return get_multi_parameter(characteristics, outputs, silent);
        }

        std::vector<information> get_all_parameters_type(const std::string& type_var)
        {
            if (type_var == "network")
                return network_parameters;
            else if (type_var == "tree")
                return tree_parameters;
            else if (type_var == "model")
                return model_parameters;
            else if (type_var == "connections")
                return connections;
            else if (type_var == "outputs")
                return outputs;
            else
            {
                std::cout << "Invalid name of parameter *" << type_var << "*." << std::endl;
                return std::vector<information>();
            }
        }

        void push_back(information& param, const std::string& type_var)
        {
            if (type_var == "network")
                network_parameters.push_back(param);
            else if (type_var == "tree")
                tree_parameters.push_back(param);
            else if (type_var == "model")
                model_parameters.push_back(param);
            else if (type_var == "connections")
                connections.push_back(param);
            else if (type_var == "outputs")
                outputs.push_back(param);
        }
    private:
        bool is_subset(const std::vector<characteristic>& A, const std::vector<characteristic>& B) const
        {
            bool equal = true;
            std::vector<characteristic> Ch1 = A;
            std::vector<characteristic> Ch2 = B;
            for (characteristic& cha : Ch1)
            {
                bool found_equal = false;
                for (characteristic& param_cha : Ch2)
                {
                    if (cha.get_name() == param_cha.get_name() && cha.get_value() == param_cha.get_value() && cha.get_values() == param_cha.get_values())
                    {
                        found_equal = true;
                        break;
                    }
                }
                if(!found_equal)
                {
                    equal = false;
                    break;
                }
            }
            return equal;
        }
        bool exist(information& param, std::vector<information>& parameters)
        {
            for (information& param_comp : parameters)
            {
                if (is_subset(param_comp.get_characteristics(), param.get_characteristics()))
                {
                    std::cout << "WARNING! information with the following characteristics already exist:" << std::endl;
                    for (characteristic& cha : param.get_characteristics())
                    {
                        std::cout << cha.get_name() << " : ";
                        if (cha.get_values().size() > 0)
                        {
                            std::cout << "[.";
                            for (value_T& val : cha.get_values()) 
                                std::cout << val << ".";
                            std::cout << "]" << std::endl;
                        }
                        else std::cout << cha.get_value() << std::endl;
                    }
                    std::cout << "Changing value *" << param_comp.get_value() <<"* to *" << param.get_value() << "*." << std::endl;
                    param.set_value(param_comp.get_value());
                    return true;
                }
            }
            return false;
        }
        std::vector<information> network_parameters;
        std::vector<information> tree_parameters;
        std::vector<information> model_parameters;
        std::vector<information> connections;
        std::vector<information> outputs;
};

class graph_data
{
    public:
        information get_info(std::vector<characteristic>& characteristics, const std::string& nameinfo)
        {
            std::vector<information> elements;
            if (nameinfo == "parameters") elements = parameters;
            else if (nameinfo == "variables") elements = variables;
            else if (nameinfo == "constraints") elements = constraints;
            else if (nameinfo == "binary variables") elements = binary_variables;
            else
            {
                std::cout << "Incorrect data type has been provided";
                return information();
            }
            for (information& info : elements)
                if (info.get_characteristics().size() == characteristics.size() && is_subset(characteristics, info.get_characteristics()))
                    return info;
            std::cout << "WARNING! information with the following characteristics not found:" << std::endl;
            for (characteristic& cha : characteristics)
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }
            return information();
        }
        std::vector<information> get_multi_info (const std::vector<characteristic>& characteristics, const std::string& nameinfo, const bool silent) const
        {
            std::vector<information> elements;
            if (nameinfo == "parameters") elements = parameters;
            else if (nameinfo == "variables") elements = variables;
            else if (nameinfo == "constraints") elements = constraints;
            else if (nameinfo == "binary variables") elements = binary_variables;
            else
            {
                std::cout << "Incorrect data type has been provided";
                return std::vector<information>();
            }
            std::vector<information> output;
            for (information& info : elements)
                if (is_subset(characteristics, info.get_characteristics()))
                    output.push_back(info);
            if (output.size() > 0) return output;

            if (!silent)
            {
                std::cout << "WARNING! information with the following characteristics not found:" << std::endl;
                for (const characteristic& cha : characteristics)
                {
                    std::cout << cha.get_name() << " : ";
                    if (cha.get_values().size() > 0)
                    {
                        std::cout << "[.";
                        for (value_T& val : cha.get_values()) 
                            std::cout << val << ".";
                        std::cout << "]" << std::endl;
                    }
                    else std::cout << cha.get_value() << std::endl;
                }
            }
            return std::vector<information>();
        }
        std::vector<information> get_all_info (const std::string& nameinfo) const
        {
            if (nameinfo == "parameters") return parameters;
            else if (nameinfo == "variables") return variables;
            else if (nameinfo == "constraints") return constraints;
            else if (nameinfo == "binary variables") return binary_variables;
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
            return std::vector<information>();
        }
        characteristic get_characteristic (const std::string& na) const
        {
            for (const characteristic& cha: characteristics)
                if (cha.get_name() == na) return cha;
            std::cout << "Characteristic with name *" << na << "* does not exist" << std::endl;
            return characteristic("N/A", std::string("N/A"), false);
        }
        std::vector<characteristic> get_characteristics() const{return characteristics;}

        void clear(const std::string& dataname)
        {
            if (dataname == "characteristics") characteristics.clear();
            else if (dataname == "parameters") parameters.clear();
            else if (dataname == "variables") variables.clear();
            else if (dataname == "binary variables") binary_variables.clear();
            else if (dataname == "all")
            {
                characteristics.clear();
                parameters.clear();
                variables.clear();
            }
            else std::cout << "*" << dataname << "* is not a valid name" << std::endl;
        }
        
        void set_characteristic(const characteristic& cha, const bool& is_vector) 
        {
            for (characteristic& element : characteristics)
            {
                if (element.get_name() == cha.get_name() && !is_vector)
                {
                    std::cout << "WARNING! Characteristic with name *" << cha.get_name() <<"* already exist with value *" << element.get_value() <<"*. The value will be replaced with *" << cha.get_value() <<"*." << std::endl;
                    element = cha;
                    return;
                }
                else if (element.get_name() == cha.get_name() && is_vector)
                {
                    element.insert(cha.get_values());
                    return;
                }
            }
            characteristics.push_back(cha);
        }
        void set_characteristics(const std::vector<characteristic>& cha){characteristics=cha;}
        void set_information(const std::vector<information>& info, std::string& name)
        {
            if (name == "parameters") parameters=info;
            else if (name == "variables") variables=info;
            else if (name == "constraints") constraints=info;
            else if (name == "binary variables") binary_variables=info;
        }
        void replace_information(const information& old_info, const information& new_info, const std::string& nameinfo)
        {
            int position;
            if (nameinfo == "parameters")
            {
                position = get_position_information(old_info, parameters);
                if (position == -1) return;
                parameters[position] = new_info;
            }
            else if (nameinfo == "variables")
            {
                position = get_position_information(old_info, variables);
                if (position == -1) return;
                variables[position] = new_info;
            }
            else if (nameinfo == "constraints")
            {
                position = get_position_information(old_info, constraints);
                if (position == -1) return;
                constraints[position] = new_info;
            }
            else if (nameinfo == "binary variables")
            {
                position = get_position_information(old_info, binary_variables);
                if (position == -1) return;
                binary_variables[position] = new_info;
            }
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
        }
        void update_value(const information& element, const std::string& nameinfo)
        {
            if (nameinfo == "parameters")
                update_info_value(element, parameters);
            else if (nameinfo == "variables")
                update_info_value(element, variables);
            else if (nameinfo == "constraints")
                update_info_value(element, constraints);
            else if (nameinfo == "binary variables")
                update_info_value(element, binary_variables);
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
        }
        void push_back(information& element, const std::string& nameinfo)
        {
            if (nameinfo == "parameters" && !exist(element, parameters))
                parameters.push_back(element);
            else if (nameinfo == "variables" && !exist(element, variables))
                variables.push_back(element);
            else if (nameinfo == "constraints" && !exist(element, constraints))
                constraints.push_back(element);
            else if (nameinfo == "binary variables" && !exist(element, binary_variables))
                binary_variables.push_back(element);
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
        }
        void push_back(information& element, const std::string& nameinfo, const characteristic& cha, const bool is_vector)
        {
            std::vector<information>* elements;
            if (nameinfo == "parameters") elements = &parameters;
            else if (nameinfo == "variables") elements = &variables;
            else if (nameinfo == "constraints") elements = &constraints;
            else if (nameinfo == "binary variables") elements = &binary_variables;
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
            
            bool found = false;
            for (information& info : *elements)
            {
                if (is_subset(element.get_characteristics(), info.get_characteristics()))
                {
                    info.set_characteristic(cha, is_vector);
                    found = true;
                    break;
                }
            }
            if (found) return;
            std::cout << "WARNING! information with the following characteristics not found:" << std::endl;
            for (const characteristic& cha : element.get_characteristics())
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }
        }
        void push_back(information& element, const std::string& nameinfo, const std::vector< std::pair<characteristic, bool> >& m_cha)
        {
            std::vector<information>* elements;
            if (nameinfo == "parameters") elements = &parameters;
            else if (nameinfo == "variables") elements = &variables;
            else if (nameinfo == "constraints") elements = &constraints;
            else if (nameinfo == "binary variables") elements = &binary_variables;
            else
                std::cout << "name of information *" << nameinfo << "* not identified" << std::endl;
            
            bool found = false;
            for (information& info : *elements)
            {
                if (is_subset(element.get_characteristics(), info.get_characteristics()))
                {
                    for (const std::pair<characteristic, bool>& cha : m_cha)
                        info.set_characteristic(cha.first, cha.second);
                    found = true;
                    break;
                }
            }
            if (found) return;
            std::cout << "WARNING! information with the following characteristics not found:" << std::endl;
            for (const characteristic& cha : element.get_characteristics())
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }
        }

        void insert(const std::string& nameinfo, const information& key, information& element)
        {
            graph_info[nameinfo][key] = element;
        }
        
        information find(const std::string& nameinfo, const information& key)
        {
            if (graph_info[nameinfo].find(key) != graph_info[nameinfo].end())
            {
                information info = graph_info[nameinfo][key];
                return info;
            }
            else
                return information();
        }

    private:
        bool is_subset(const std::vector<characteristic>& A, const std::vector<characteristic>& B) const
        {
            bool equal = true;
            std::vector<characteristic> Ch1 = A;
            std::vector<characteristic> Ch2 = B;
            for (characteristic& cha : Ch1)
            {
                bool found_equal = false;
                for (characteristic& param_cha : Ch2)
                {
                    if (cha.get_name() == param_cha.get_name() && cha.get_value() == param_cha.get_value() && cha.get_values() == param_cha.get_values())
                    {
                        found_equal = true;
                        break;
                    }
                }
                if(!found_equal)
                {
                    equal = false;
                    break;
                }
            }
            return equal;
        }
        bool exist(const information& info, std::vector<information>& input) const
        {
            for (information& in : input)
            {
                if (is_subset(in.get_characteristics(), info.get_characteristics()))
                {
                    std::cout << "WARNING! information with the following characteristics already exist:" << std::endl;
                    for (characteristic& cha : info.get_characteristics())
                    {
                        std::cout << cha.get_name() << " : ";
                        if (cha.get_values().size() > 0)
                        {
                            std::cout << "[.";
                            for (value_T& val : cha.get_values()) 
                                std::cout << val << ".";
                            std::cout << "]" << std::endl;
                        }
                        else std::cout << cha.get_value() << std::endl;
                    }
                    std::cout << "Changing value *" << in.get_value() <<"* to *" << info.get_value() << "*." << std::endl;
                    in.set_value(info.get_value());
                    return true;
                }
            }
            return false;
        }
        void update_info_value(const information& info, std::vector<information>& input)
        {
            for (information& in : input)
            {
                if (is_subset(in.get_characteristics(), info.get_characteristics()))
                {
                    in.set_value(info.get_value());
                    return;
                }
            }
            std::cout << "WARNING! information with the following characteristics does not exist:" << std::endl;
            for (characteristic& cha : info.get_characteristics())
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }
        }
        int get_position_information(const information& info, std::vector<information>& input)
        {
            int counter = 0;
            for (information& in : input)
            {
                if (is_subset(in.get_characteristics(), info.get_characteristics()))
                    return counter;
                counter++;
            }
            std::cout << "WARNING! information with the following characteristics does not exist:" << std::endl;
            for (characteristic& cha : info.get_characteristics())
            {
                std::cout << cha.get_name() << " : ";
                if (cha.get_values().size() > 0)
                {
                    std::cout << "[.";
                    for (value_T& val : cha.get_values()) 
                        std::cout << val << ".";
                    std::cout << "]" << std::endl;
                }
                else std::cout << cha.get_value() << std::endl;
            }
            return -1;
        }
        std::vector<characteristic> characteristics;
        // info:
        std::vector<information> parameters;
        std::vector<information> variables;
        std::vector<information> constraints;
        std::vector<information> binary_variables;
        boost::unordered_map<std::string, boost::unordered_map<information, information> > graph_info;
};

class matrix_representation
{
    public:
        matrix_representation()
        {
            number_variables = 0;
            number_constraints = 0;
        }
        int add_variable(const double& upper, const double& lower, const double& cost)
        {
            colLower.push_back(lower);
            colUpper.push_back(upper);
            objective.push_back(cost);
            return int(colLower.size()) - 1;
        }
        int add_constraint(const double& upper, const double& lower)
        {
            rowLower.push_back(lower);
            rowUpper.push_back(upper);
            return int(rowLower.size()) - 1;
        }
        void add_value2matrix(const double& val, const int& row, const int& column)
        {
            elements.push_back(val);
            rows.push_back(row);
            columns.push_back(column);
        }
        void add_active(const std::string& name, const int& column_or_row)
        {
            if (name == "row") active_rows.push_back(column_or_row);
            if (name == "column") active_columns.push_back(column_or_row);
        }
        void update_variable(const int column, const double& upper, const double& lower, const double& cost)
        {
            colLower[column] = lower;
            colUpper[column] = upper;
            objective[column] = cost;
        }
        void update_constraint(const int row, const double& upper, const double& lower)
        {
            rowLower[row] = lower;
            rowUpper[row] = upper;
        }
        int modify_coefficient(int row, int column, double newElement)
        {
            std::vector<int>::iterator it_rows = std::find(rows.begin(), rows.end(), row);
            if (it_rows == rows.end())
            {
                std::cout << "WARNING! row *" << row <<"* is not in matrix of coefficients." << std::endl;
                return -12;
            }
            int position = it_rows - rows.begin();
            for (int pos = position; pos < int(columns.size()); pos++)
            {
                if (columns[pos] == column)
                {
                    position = pos;
                    break;
                }    
                else if (pos == int(columns.size()) - 1 || rows[pos] != row)
                {
                    std::cout << "WARNING! column *" << column <<"* is not in matrix of coefficients." << std::endl;
                    return -12;
                }
            }
            elements[position] = newElement;
            return 0;
        }

        int size(const std::string& name) const
        {
            if (name=="variables") return int(colLower.size());
            else if (name=="constraints") return int(rowLower.size());
            else if (name=="coefficients") return int(elements.size());
            else if (name=="active constraints") return int(active_rows.size());
            else if (name=="active variables") return int(active_columns.size());
            else
                std::cout << "Invalid name *" << name << "* for problem information" << std::endl;
            return -1;
        }
        int get_position_matrix(const std::string& name, const int pos) const
        {
            if (name == "column") return columns[pos];
            else if (name == "row") return rows[pos];
            else
                std::cout << "Invalid name *" << name << "*. " << std::endl;
            return -1;
        }
        double get_value(const std::string& name, const int pos) const
        {
            if (name == "coefficient") return elements[pos];
            else if (name == "cost") return objective[pos];
            else if (name == "variable lower") return colLower[pos];
            else if (name == "variable upper") return colUpper[pos];
            else if (name == "constraint lower") return rowLower[pos];
            else if (name == "constraint upper") return rowUpper[pos];
            else
                std::cout << "Invalid name *" << name << "*. " << std::endl;
            return 0.0;
        }
        std::vector<int> get_positions(const std::string& name) const
        {
            if (name == "columns") return columns;
            else if (name == "rows") return rows;
            else if (name == "active constraints") return active_rows;
            else if (name == "active variables") return active_columns;
            else
                std::cout << "Invalid name *" << name << "*. " << std::endl;
            return std::vector<int>();
        }
        std::vector<double> get_values(const std::string& name) const
        {
            if (name == "coefficients") return elements;
            else if (name == "cost") return objective;
            else if (name == "variables lower") return colLower;
            else if (name == "variables upper") return colUpper;
            else if (name == "constraints lower") return rowLower;
            else if (name == "constraints upper") return rowUpper;
            else
                std::cout << "Invalid name *" << name << "*. " << std::endl;
            return std::vector<double>();
        }
        std::pair<double, int> get_coefficient(int row, int column)
        {
            std::vector<int>::iterator it_rows = std::find(rows.begin(), rows.end(), row);
            if (it_rows == rows.end())
            {
                std::cout << "WARNING! row *" << row <<"* is not in matrix of coefficients." << std::endl;
                return std::make_pair(0.0, -12);
            }
            int position = it_rows - rows.begin();
            for (int pos = position; pos < int(columns.size()); pos++)
            {
                if (columns[pos] == column)
                    return std::make_pair(elements[pos], 0);
                else if (pos == int(columns.size()) - 1 || rows[pos] != row)
                {
                    std::cout << "WARNING! column *" << column <<"* is not in matrix of coefficients." << std::endl;
                    return std::make_pair(0.0, -12);
                }
            }
        }
    private:
        std::vector<int> columns;
        std::vector<int> rows;
        std::vector<double> elements;
        std::vector<double> objective;
        std::vector<double> rowLower;
        std::vector<double> rowUpper;
        std::vector<double> colLower;
        std::vector<double> colUpper;
        std::vector<int> active_rows;
        std::vector<int> active_columns;
        int number_variables;
        int number_constraints;
};

typedef boost::adjacency_list<  boost::vecS, boost::vecS, 
                                boost::bidirectionalS,
                                graph_data,
                                graph_data
                                > GraphType;

typedef boost::graph_traits<GraphType>::adjacency_iterator AdjacencyIterator;
//Edge iterator for or graph
typedef boost::graph_traits<GraphType>::edge_iterator edge_iterator;
typedef boost::graph_traits<GraphType>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<GraphType>::in_edge_iterator in_edge_iterator;

class graphs
{
    public:
        size_t number_nodes(const std::string& graphname)
        {
            if (graphname == "tree") return boost::num_vertices(tree_data);
            else if (graphname == "network") return boost::num_vertices(network_data);
            else
            {
                std::cout << "Invalid graph name *" << graphname << "*." << std::endl;
                return 0;
            }
        }
        size_t number_edges(const std::string& graphname)
        {
            if (graphname == "tree") return boost::num_edges(tree_data);
            else if (graphname == "network") return boost::num_edges(network_data);
            else
            {
                std::cout << "Invalid graph name *" << graphname << "*." << std::endl;
                return 0;
            }
        }
        graph_data get(const std::string& graphname, const int& pos, const std::string& component)
        {
            if (graphname == "tree" && component == "vertex") return tree_data[pos];
            else if (graphname == "network" && component == "vertex") return network_data[pos];
            else if (graphname == "tree" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(tree_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                        return tree_data[*ei];
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else if (graphname == "network" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(network_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                        return network_data[*ei];
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else
                std::cout << "Invalid graph name *" << graphname << "* and/or invalid component *" << component << "*." << std::endl;
        }
        GraphType get_grafo(const std::string& graphname)
        {
            if (graphname == "tree") return tree_data;
            else if (graphname == "network") return network_data;
            else
                std::cout << "Invalid graph name *" << graphname << "*." << std::endl;
            return GraphType();
        }
        std::vector<int> get_positions(const std::string& graphname, const std::string& type)
        {
            if (graph_elements_positions[graphname].find(type) == graph_elements_positions[graphname].end()) return std::vector<int>();
            return graph_elements_positions[graphname][type];
        }

        void add_vertex(const std::string& graphname, const graph_data& node)
        {
            if (graphname == "tree") boost::add_vertex(node, tree_data);
            else if (graphname == "network") 
            {
                graph_elements_positions[graphname][boost::get<std::string>(node.get_characteristic("type").get_value())].push_back(boost::num_vertices(network_data));
                boost::add_vertex(node, network_data);
            }
        }
        void add_edge(const std::string& graphname, const int& A, const int &B, const graph_data& branch)
        {
            if (graphname == "tree") boost::add_edge(A, B, branch, tree_data);
            else if (graphname == "network") boost::add_edge(A, B, branch, network_data);
        }
        void set(const std::string& graphname, const int& pos, const graph_data& data, const std::string& component)
        {
            if (graphname == "tree" && component == "vertex") tree_data[pos] = data;
            else if (graphname == "network" && component == "vertex") network_data[pos] = data;
            else if (graphname == "tree" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(tree_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        tree_data[*ei] = data;
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else if (graphname == "network" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(network_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        network_data[*ei] = data;
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else
                std::cout << "Invalid graph name *" << graphname << "* and/or invalid component *" << component << "*." << std::endl;
        }

        void insert2component(const std::string& graphname, const int& pos, const std::string& component, const std::string& nameinfo, const information& key, information& element)
        {
            if (graphname == "network" && component == "vertex") network_data[pos].insert(nameinfo, key, element);
        }
        information find_component_information(const std::string& graphname, const int& pos, const std::string& component, const std::string& nameinfo, const information& key)
        {
            if (graphname == "network" && component == "vertex") return network_data[pos].find(nameinfo, key);
        }
    
        void push_back_info(const std::string& graphname, const int& pos, const std::string& component, information& element, const std::string& nameinfo, const std::vector< std::pair<characteristic, bool> >& m_cha)
        {
            if (graphname == "tree" && component == "vertex") tree_data[pos].push_back(element, nameinfo, m_cha);
            else if (graphname == "network" && component == "vertex") network_data[pos].push_back(element, nameinfo, m_cha);
            else if (graphname == "tree" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(tree_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        tree_data[*ei].push_back(element, nameinfo, m_cha);
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else if (graphname == "network" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(network_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        network_data[*ei].push_back(element, nameinfo, m_cha);
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else
                std::cout << "Invalid graph name *" << graphname << "* and/or invalid component *" << component << "*." << std::endl;
        }

        void push_back_info(const std::string& graphname, const int& pos, const std::string& component, information& element, const std::string& nameinfo)
        {
            if (graphname == "tree" && component == "vertex") tree_data[pos].push_back(element, nameinfo);
            else if (graphname == "network" && component == "vertex") network_data[pos].push_back(element, nameinfo);
            else if (graphname == "tree" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(tree_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        tree_data[*ei].push_back(element, nameinfo);
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else if (graphname == "network" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(network_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                    {
                        network_data[*ei].push_back(element, nameinfo);
                        return;
                    }
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else
                std::cout << "Invalid graph name *" << graphname << "* and/or invalid component *" << component << "*." << std::endl;
        }
    
        std::vector<information> get_multi_info (const std::string& graphname, const int& pos, const std::string& component, const std::vector<characteristic>& characteristics, const std::string& nameinfo, const bool silent) const
        {
            if (graphname == "tree" && component == "vertex") return tree_data[pos].get_multi_info(characteristics, nameinfo, silent);
            else if (graphname == "network" && component == "vertex") return network_data[pos].get_multi_info(characteristics, nameinfo, silent);
            else if (graphname == "tree" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(tree_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                        return tree_data[*ei].get_multi_info(characteristics, nameinfo, silent);
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else if (graphname == "network" && component == "edge")
            {
                int counter = 0;
                edge_iterator ei,ei_end;
                for (boost::tie(ei, ei_end) = boost::edges(network_data); ei != ei_end; ++ei)
                {
                    if (counter == pos)
                        return network_data[*ei].get_multi_info(characteristics, nameinfo, silent);
                    else counter++;
                }
                std::cout << "Invalid edge position *" << pos << "* for graph *" << graphname << "*." << std::endl;
            }
            else
                std::cout << "Invalid graph name *" << graphname << "* and/or invalid component *" << component << "*." << std::endl;
        }
    
    private:
        GraphType tree_data;
        GraphType network_data;
        boost::unordered_map<std::string, boost::unordered_map<std::string, std::vector<int> > > graph_elements_positions;
};

class clp_model
{
    public:
        void load_model(const matrix_representation& problem_matrix)
        {
            CoinPackedMatrix matrix(true, problem_matrix.get_positions("rows").data(), problem_matrix.get_positions("columns").data(), problem_matrix.get_values("coefficients").data(), problem_matrix.size("coefficients"));

            model.loadProblem(matrix, problem_matrix.get_values("variables lower").data(), problem_matrix.get_values("variables upper").data(), problem_matrix.get_values("cost").data(), problem_matrix.get_values("constraints lower").data(), problem_matrix.get_values("constraints upper").data());

            active_rows = problem_matrix.get_positions("active constraints");
            active_columns = problem_matrix.get_positions("active variables");
        }
        void setColBounds (int elementIndex, double newlower, double newupper) {model.setColumnBounds(elementIndex, newlower, newupper);}
        void setRowBounds (int elementIndex, double newlower, double newupper) {model.setRowBounds(elementIndex, newlower, newupper);}
        void setObjCoeff (int elementIndex, double newcost) {model.setObjCoeff(elementIndex, newcost);}
        void modifyCoefficient (int row, int column, double newElement, bool keepZero=false)
        {
            model.modifyCoefficient(row, column, newElement, keepZero);
        }

        ClpSimplex get_model() const {return model;}
        void solve(const std::string& name_method)
        {

            ClpSimplex reduced_model(&model, active_rows.size(), active_rows.data(), active_columns.size(), active_columns.data());
            /* +1 to minimize, -1 to maximize, and 0 to ignore */
            reduced_model.setOptimizationDirection(1);
            /*
            Amount of print out:
                0 - none
                1 - just final
                2 - just factorizations
                3 - as 2 plus a bit more
                4 - verbose
            */
            reduced_model.setLogLevel(1);
            if(name_method == "primal") reduced_model.primal();
            else if(name_method == "dual") reduced_model.dual();
            else if(name_method == "initial solve")
            {
                ClpSolve solveOptions;
                solveOptions.setSolveType(ClpSolve::usePrimal);
                reduced_model.initialSolve(solveOptions);
            }
            double * solution = model.primalColumnSolution();
            const double * smallSolution = reduced_model.primalColumnSolution();
            for (size_t j = 0; j < active_columns.size(); j++) {
                 solution[active_columns[j]] = smallSolution[j];
                 model.setColumnStatus(active_columns[j], reduced_model.getColumnStatus(j));
            }
            for (size_t iRow = 0; iRow < active_rows.size(); iRow++) {
                 model.setRowStatus(active_rows[iRow], reduced_model.getRowStatus(iRow));
            }
        }
        std::vector<double> get_solution()
        {
            double *sol = model.primalColumnSolution();
            return(std::vector<double>(sol, sol+model.getNumCols()));
        }
    private:
        ClpSimplex  model;
        double objective_function;
        std::vector<int> active_rows;
        std::vector<int> active_columns;
};

class MOEA
{
    public:
        void create_data_set(const std::string& name_information)
        {
            data[name_information] = std::vector<information>();
        }
        void push_back(const std::string& name_information, const information& info)
        {
            std::map<std::string, std::vector<information> >::iterator it = data.find(name_information);
            if (it != data.end() && !exist_in_set(info, it -> second))
                it -> second.push_back(info);
            else if (it == data.end())
                std::cout << "WARNING! Information with name *" << name_information << "* does not exist in the MOEA class." << std::endl;
            else
            {
                std::cout << "WARNING! Information with characteristics the following characteristics already exist:" << std::endl;
                for (characteristic& cha : info.get_characteristics())
                {
                    std::cout << cha.get_name() << " : ";
                    if (cha.get_values().size() > 0)
                    {
                        std::cout << "[.";
                        for (value_T& val : cha.get_values()) 
                            std::cout << val << ".";
                        std::cout << "]" << std::endl;
                    }
                    else std::cout << cha.get_value() << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::vector<information> get_data_set(const std::string& name_information)
        {
            std::map<std::string, std::vector<information> >::iterator it = data.find(name_information);
            if (it != data.end())
                return data[name_information];
            else if (it == data.end())
                std::cout << "WARNING! Information with name *" << name_information << "* does not exist in the MOEA class." << std::endl;
            return std::vector<information>();
        }
    private:
        bool is_subset(const std::vector<characteristic>& A, const std::vector<characteristic>& B) const
        {
            bool equal = true;
            std::vector<characteristic> Ch1 = A;
            std::vector<characteristic> Ch2 = B;
            for (characteristic& cha : Ch1)
            {
                bool found_equal = false;
                for (characteristic& param_cha : Ch2)
                {
                    if (cha.get_name() == param_cha.get_name() && cha.get_value() == param_cha.get_value() && cha.get_values() == param_cha.get_values())
                    {
                        found_equal = true;
                        break;
                    }
                }
                if(!found_equal)
                {
                    equal = false;
                    break;
                }
            }
            return equal;
        }
        bool exist_in_set(const information& info, std::vector<information>& input) const
        {
            for (information& in : input)
                if (is_subset(in.get_characteristics(), info.get_characteristics()))
                    return true;
            return false;
        }
        
        std::map<std::string, std::vector<information> > data;
};

class models{

    public:

        models();
        ~models();

        void create_parameter();
        void load_double(const std::string& na, const double& val, const bool& is_vector);
        void load_integer(const std::string& na, const int& val, const bool& is_vector);
        void load_bool(const std::string& na, const bool& val, const bool& is_vector);
        void load_string(const std::string& na, const std::string& val, const bool& is_vector);
        void set_parameter(const std::string& typ);
        int update_parameter();
        void get_MOEA_variables(std::vector<std::string>& IDs, std::vector<std::string>& names, std::vector<double>& min, std::vector<double>& max);
        void get_moea_objectives(std::vector<std::string>& names);

        void return_outputs(std::vector<double>& values, std::vector<int>& starts, std::vector< std::vector< std::vector< std::string> > >& characteristics);

        void initialise();
        void evaluate();

    private:

        const double PENALTY = 1000000.0;
        const double TOLERANCE_REACTANCE = 1e-8;

        information candidate;
        parameters data_parameters;
        graphs grafos;
        matrix_representation problem_matrix;
        clp_model solver;
        MOEA moea_model;

        void load_parameter(const std::string& na, const value_T& val, const bool& is_vector);

        void create_graph_databases();
        void create_nodes_network();
        void create_edges_network();
        void create_nodes_tree();
        void create_edges_tree();
        std::vector<characteristic> levels_tree();
        std::vector< std::vector<characteristic> > names_levels_tree();

        int create_problem_element(const std::string& name_var, const double max, const double min, std::vector<std::pair<characteristic, bool> >& extra_characteristics, const int node_graph, const std::string& name_graph, const std::string& graph_component, const std::string& type_info, const double cost);

        std::pair<information, int> get_information(const graph_data &vertex, const std::string& name_info, const std::vector<characteristic>& extra_characteristics, const std::string& name_datainfo, bool silent, const int pos=-1, const std::string name_graph = "", const std::string component = "");

        std::pair<std::vector<std::vector<double> >, int> create_dc_opf_susceptance_matrix(const std::vector<value_T>& pt, const double hour);

        int recursive_balance_tree_search(information& info, std::vector<information>& storage);
        void update_parameters_references_tree(const std::vector<std::string>& references, const std::string& parameter_name);
        int create_balance_tree_model();
        std::pair<std::vector<std::string>, int> connections_energy_tree_model();
        int declare_balance_tree_variables(const std::string& id);
        int declare_balance_tree_constraints(const std::string& id);
        int create_balance_tree_matrix(const std::string& id);

        int create_dc_opf_model();
        int convert2per_unit();
        std::pair<std::vector<std::pair<characteristic, bool>>, int> piecewise_linearisation(information& info, const value_T max, const value_T min);
        std::vector< std::pair<value_T, value_T> > breakpoints_piecewise_linearisation(const information& info, const value_T max, const value_T min);
        int declare_dc_opf_variables(const std::vector<value_T>& pt, const double hour);
        int declare_dc_opf_constraints(const std::vector<value_T>& pt, const double hour);
        int connections_parameters_variables_dc_opf(const information& subscripts);

        int get_position_matrix_dc_opf(const std::string& name_constraint, graph_data& vertex, const std::vector<characteristic>& characteristics, const std::string& var_con);

        std::pair< int, std::vector<information> > create_dc_opf_matrix(const std::vector<information>& subscripts);

        int declare_dc_opf_tree_links_constraints(const std::vector<value_T>& pt);
        int create_dc_opf_tree_links_matrix(const std::vector<value_T>& pt);
        int update_cost_variables_constraints_dc_opf_tree_links(const std::vector<value_T>& pt);
        int create_dc_opf_tree_links();

        int update_bnds_obj(graph_data data, const std::string& name_graph, const std::string& component, const int position_in_graph, const std::string& name_info, const std::string& name_datainfo, const std::vector<characteristic>& extra_characteristics, bool silent, value_T new_max, value_T new_min, value_T new_cost, const bool clp_change);

        void store_solution();

        void accumulate_unique_characteristics_outputs(std::vector<information>& information_required, const information& output, const std::string& name);

        int create_moea_problem();
        int declare_moea_variables();
        int declare_moea_objectives();

};