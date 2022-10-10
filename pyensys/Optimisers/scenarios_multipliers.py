
import os
import json


def get_mult(country_selected= "HR"):
    # abspath = os.path.abspath(__file__)
    # dname = os.path.dirname(abspath)
    # os.chdir(dname)
    
    # Select the case study: "PT", "UK" or "HR"
    # country_selected = "HR" 
    
    # Select if scenarios should be plotted:
    plot_scenarios = False
    
    # Input data: mean annual growth (%) - available in D2.3 report
    
    if country_selected == "PT": # PORTUGUESE POWER SYSTEM:
        mean_annual_growth = {
            "Active":{
                "2030":1.69,
                "2040":2.29,
                "2050":1.87
            },
            "Slow":{
                "2030":0.66,
                "2040":1.32,
                "2050":1.16
            }
        }
    elif country_selected == "UK":# UK CASE:
            ### Initial forecast from deliverables D2.3:
        # mean_annual_growth = {
        #     "Active":{
        #         "2030":0.8,
        #         "2040":1.6,
        #         "2050":1.0
        #     },
        #     "Slow":{
        #         "2030":0.45,
        #         "2040":1.08,
        #         "2050":1.2
        #     }
        ### Updated forecast to match FES (increased loads):
            mean_annual_growth = {
            "Active":{
                "2030":1.89,
                "2040":3.0,
                "2050":2.5,
            },
            "Slow":{
                "2030":1.1,
                "2040":2.0,
                "2050":1.0
            }
        }
    elif country_selected == "HR":# CROATIAN POWER SYSTEM:
        mean_annual_growth = {
            "Active":{
                "2030":1.6,
                "2040":1.1,
                "2050":2.6
            },
            "Slow":{
                "2030":1.35,
                "2040":0.65,
                "2050":0.7
            }
        }

    # Output data: defining the demand multipliers for years and scenarios
    # (relative values compared to the year 2020)
    mult = {"2020":[1],
            "2030":[None]*2,
            "2040":[None]*4,
            "2050":[None]*8
    }
    
    # Year 2030: defining 2 scenarios, Active (A) and Slow (S)
    mult["2030"][0] = 1 + mean_annual_growth["Active"]["2030"]*10/100
    mult["2030"][1] = 1 + mean_annual_growth["Slow"]["2030"]*10/100
    
    # Year 2040: defining 4 scenarios, AA, AS, SA, and SS
    mult["2040"][0] = mult["2030"][0]*(1 + mean_annual_growth["Active"]["2040"]*10/100)
    mult["2040"][3] = mult["2030"][1]*(1 + mean_annual_growth["Slow"]["2040"]*10/100)
    mult_year_range = mult["2040"][0] - mult["2040"][3] # forecast range
    mult["2040"][1] = mult["2040"][0] - mult_year_range*1/3
    mult["2040"][2] = mult["2040"][0] - mult_year_range*2/3
    
    # Year 2050: defining 8 scenarios, AAA, AAS, ASA, ASS, SAA, SAS, SSA, and SSS
    mult["2050"][0] = mult["2040"][0]*(1 + mean_annual_growth["Active"]["2050"]*10/100)
    mult["2050"][7] = mult["2040"][1]*(1 + mean_annual_growth["Slow"]["2050"]*10/100)
    mult_year_range = mult["2050"][0] - mult["2050"][7] # forecast range
    for interim_sc in range(6): # compute the multipliers for interim scenarios within the forecast range
        mult["2050"][interim_sc + 1] = mult["2050"][0] - mult_year_range*(interim_sc + 1)/7
    
    # print()
    # print("The multipliers for the scenarios:")
    # print(mult)
    
    if plot_scenarios == True:
        from matplotlib import pyplot as plt
        # plt.grid()

        # Main scenarios given by D2.3 (Active and Slow economies):
        plt.plot(["2020", "2030"],[mult["2020"][0],mult["2030"][0]], color = "black")
        plt.plot(["2020", "2030"],[mult["2020"][0],mult["2030"][1]], color = "black")
        plt.plot(["2030", "2040"],[mult["2030"][0],mult["2040"][0]], color = "black")
        plt.plot(["2030", "2040"],[mult["2030"][-1],mult["2040"][-1]], color = "black")
        plt.plot(["2040", "2050"],[mult["2040"][0],mult["2050"][0]], color = "black")
        plt.plot(["2040", "2050"],[mult["2040"][-1],mult["2050"][-1]], color = "black")
        plt.scatter("2020", mult["2020"][0], color = "black")
        plt.scatter(["2030", "2040", "2050"], [mult["2030"][0], mult["2040"][0], mult["2050"][0]], color = "black")
        plt.scatter(["2030", "2040", "2050"], [mult["2030"][-1], mult["2040"][-1], mult["2050"][-1]], color = "black")
        
        
        # Interim scenarios (between Active and Slow):
        plt.plot(["2030", "2040"],[mult["2030"][0],mult["2040"][1]], color = "grey")
        plt.plot(["2030", "2040"],[mult["2030"][-1],mult["2040"][2]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][0],mult["2050"][1]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][1],mult["2050"][2]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][1],mult["2050"][3]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][2],mult["2050"][4]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][2],mult["2050"][5]], color = "grey")
        plt.plot(["2040", "2050"],[mult["2040"][-1],mult["2050"][6]], color = "grey")
        plt.scatter(["2040"]*2, mult["2040"][1:3], color = "grey")
        plt.scatter(["2050"]*6, mult["2050"][1:7], color = "grey")

        plt.title("Scenarios and multipliers for the "+country_selected+" case")
        # plt.axis('square')


        plt.show()
        # plt.savefig('scenarios_multipliers.png')
        
        
        
        
    # remove keys of the dict
    multiplier = []
    aux = [2020,2030,2040,2050]
    for i in range(4):
        multiplier.append(mult.pop(str(aux[i]), None))

    print('multiplier: ')
    print(multiplier)
   
        
   
    return multiplier




