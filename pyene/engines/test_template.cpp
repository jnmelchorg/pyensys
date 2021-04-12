#include <vector>
#include <string>
#include <iostream>

using namespace std;

// New implementation

struct characteristic
{
    string name;                    // name of characteristic
    template <class T>
    static T value;                 // value of characteristic
};

struct tree_position
{
    int level;                                 // level of the tree position
    string name_level;                         // name of level
};


struct parameter
{
    vector<characteristic> characteristics;
};

void main()
{
    parameter param;
    characteristic cha;
    cha.name = "something";
    vector<tree_position> treep;
    tree_position tp;
    tree_position tp1;
    tp.level = 1;
    tp.name_level = "week";
    tp1.level = 2;
    tp1.name_level = "weekday";
    treep.push_back(tp);
    treep.push_back(tp1);
    cha.value<vector<tree_position>> = treep;
    param.characteristics.push_back(cha);

    // cout << param.characteristics[0].name << endl;
    // cout << param.characteristics[0].value << endl;
}