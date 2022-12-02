import pandas as pd, numpy as np, os, matplotlib.pyplot as plt

def calc_rsme(data_pred:list, data_act:list):
    "Takes two list like objects of equal length and calculates the Root Mean Square Error"
    #Try to calculate SE by leveraging numpy's built in ability to perform operations across lists - then reduce to RSME
    #Function accomplishes 1A
    try:
        if len(data_pred) != len(data_act):
            raise Exception("Lists must be equal")
        error_list = np.subtract(data_pred,data_act)
        se = np.sum(np.power(error_list,2))
        rmse = np.sqrt((1/len(error_list))*se)
        return rmse
    except Exception as e:
        return e

def calc_mae(data_pred:list, data_act:list):
    "Takes two list like objects of equal length and calculates Mean Absolute Error"
    #Try to calculate MAE by leveraging numpy's built in ability to perform operation across lists.
    #Function accomplishes 2A
    try:
        if len(data_pred) != len(data_act):
            raise Exception("Lists must be equal")
        error_list = np.subtract(data_pred,data_act)
        mae = (1/len(error_list))*np.sum(np.abs(error_list))
        return mae
    except Exception as e:
        return e

def bin_acc(data_pred:list, data_act:list):
    "Takes two list like objects of equal length and gets the prediction accuracy"
    #Try to calculate a binary accuracy by comparing the items literally. For the context of the CSV this works, but a more complex list would need a more robust solution.
    #This takes place by using a list comprehension for readability and speed. Afterwards use np.average to get accuracy. Accomplishes 3A
    try:
        if len(data_pred) != len(data_act):
            raise Exception("Lists must be equal")
        result_list = [1 if pred == act else 0 for pred,act in zip(data_pred,data_act)]
        accuracy = np.average(result_list)
        return accuracy
    except Exception as e:
        return e




if __name__ == "__main__":
    #Get path to python file and change directory
    cur_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_path)

    #Open housing data then run RSME and MAE functions, accomplishing 1B and 2B respectively, and print results.
    house_data = pd.read_csv("housing_data.csv")
    print(f'1B RMSE: {calc_rsme(house_data["sale_price_pred"], house_data["sale_price"])} | 2B MAE: {calc_mae(house_data["sale_price_pred"], house_data["sale_price"])}' )

    #Open mushroom data then run the accuracy function. Accomplishing 3B
    mushroom_data = pd.read_csv("mushroom_data.csv")
    print(f'3B Accuracy: {bin_acc(mushroom_data["predicted"], mushroom_data["actual"])}')

    #Generate a graph for the function outlined in 4A. Since the code in question is used to easily show the section where Y is close to 0 for convenience, I have saved
    #A full view image in the directory to demonstrate the code works.

    #Generate equally spaced p values for the graph
    p_var = np.linspace(-1,1, 100)
    y = (0.005*p_var)**6 - (0.27*p_var)**5 + (5.998*p_var)**4 - (69.919*p_var)**3 + (449.17*p_var)**2 - (1499.7*p_var) + 2028

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(p_var,y, ',:c')
    plt.xlabel("P")
    plt.ylabel("Error")
    plt.ylim(top=0.05,bottom=0)
    plt.xlim(left=0.6,right=0.601)
    plt.show()
    #4B & 4C: Based on the scrubbing on this plot I determined that the approximate P value to minimize the error (down to ~0.000005) is 6.005212.